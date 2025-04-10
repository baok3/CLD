import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader, ConcatDataset
from PIL import Image
import os
import tempfile
import random
import shutil
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Paths
BASE_DATASET = "E:\App data\data_optimize\Basee_data"
RARE_DATASET = "E:\App data\data_optimize\Rare data"
TEMP_DIR = tempfile.mkdtemp()

# EfficientNet feature extractor with projection layer to match embedding size
class EfficientNetWithProjection(nn.Module):
    def __init__(self, model, output_dim=1024):
        super(EfficientNetWithProjection, self).__init__()
        self.model = model
        self.projection = nn.Linear(1280, output_dim)  # Projection layer to 1024
    
    def forward(self, x):
        x = self.model(x)  # Get embeddings from EfficientNet
        return self.projection(x)  # Project to 1024 dimensions

# EfficientNet feature extractor
@st.cache_resource
def load_model():
    model = models.efficientnet_b0(weights="IMAGENET1K_V1")  # Load pretrained weights
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 5)  # Final output for 5 classes
    model.load_state_dict(torch.load("E:\Model save\Deep_learning_model\model\efficientnet_coffee (1).pth", map_location=DEVICE), strict=False)
    model.classifier = nn.Identity()  # Remove classifier
    model = EfficientNetWithProjection(model)  # Add projection layer to reduce dimension
    model.to(DEVICE)
    model.eval()
    return model

# Image transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Reset Model Function
def reset_model():
    # Load the EfficientNet model with pretrained weights
    model = models.efficientnet_b0(weights="IMAGENET1K_V1")  # Load pretrained weights
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 5)  # Modify for 5 classes
    model.load_state_dict(torch.load("E:\Model save\Deep_learning_model\model\efficientnet_coffee (1).pth", map_location=DEVICE), strict=False)
    model.to(DEVICE)
    model.eval()  # Ensure model is in evaluation mode after reset
    return model

# Create episode for Few-Shot Learning
def create_episode(dataset, class_indices, n_way=6, n_shot=5, n_query=5):
    selected_classes = random.sample(list(class_indices.keys()), n_way)
    support_imgs, query_imgs, support_labels, query_labels = [], [], [], []
    
    for new_label, class_id in enumerate(selected_classes):
        indices = class_indices[class_id]
        
        # Skip classes that don't have enough samples for support and query sets
        if len(indices) < n_shot + n_query:
            print(f"Skipping class {class_id} (not enough samples).")
            continue
        
        sampled_indices = random.sample(indices, n_shot + n_query)
        
        # Split the sampled indices into support and query sets
        support_imgs += [dataset[i][0] for i in sampled_indices[:n_shot]]
        query_imgs += [dataset[i][0] for i in sampled_indices[n_shot:]]
        
        support_labels += [new_label] * n_shot
        query_labels += [new_label] * n_query
    
    # Ensure that we have the correct number of support and query images
    if len(support_imgs) != n_way * n_shot or len(query_imgs) != n_way * n_query:
        raise ValueError(f"Generated episode does not have the expected number of samples. "
                         f"Support images: {len(support_imgs)}, Query images: {len(query_imgs)}")
    
    return (torch.stack(support_imgs).to(DEVICE),
            torch.tensor(support_labels).to(DEVICE),
            torch.stack(query_imgs).to(DEVICE),
            torch.tensor(query_labels).to(DEVICE))

# Loss function for Few-Shot Learning
def proto_loss(support, query, n_way, n_shot, n_query):
    if support.size(0) == 0 or query.size(0) == 0:
        return torch.tensor(0.0, requires_grad=True).to(DEVICE)
    
    # Ensure that support and query embeddings have the same dimension
    assert support.size(-1) == query.size(-1), f"Embedding dimension mismatch: support {support.size(-1)}, query {query.size(-1)}"
    
    # Check if the number of elements in support is compatible with n_way * n_shot
    num_elements = support.size(0)
    expected_size = n_way * n_shot
    if num_elements % expected_size != 0:
        raise ValueError(f"Cannot reshape support tensor. Expected number of elements: {expected_size}, but got {num_elements}.")
    
    # Reshape support to match n_way, n_shot, embedding_size
    support = support.view(n_way, n_shot, -1)
    
    # Calculate the prototypes
    prototypes = support.mean(dim=1)  # Mean across shots
    
    # Calculate the distances between the query embeddings and prototypes
    distances = torch.cdist(query, prototypes)
    labels = torch.arange(n_way).repeat_interleave(n_query).to(DEVICE)
    
    return F.cross_entropy(-distances, labels)

# Recalculate prototypes after training
@st.cache_data
def calculate_prototypes(_model, _dataset):
    loader = DataLoader(_dataset, batch_size=32, shuffle=False)
    embeddings, labels = [], []
    with torch.no_grad():
        for imgs, labs in loader:
            imgs = imgs.to(DEVICE)
            emb = _model(imgs)  # Get embeddings from the trained model
            embeddings.append(emb.cpu())
            labels.append(labs)
    embeddings = torch.cat(embeddings)
    labels = torch.cat(labels)
    prototypes = []
    
    # Calculate the prototypes (mean of embeddings per class)
    for label in range(labels.max().item() + 1):
        class_emb = embeddings[labels == label]
        prototypes.append(class_emb.mean(dim=0) if len(class_emb) > 0 else torch.zeros(embeddings[0].shape))
    
    return torch.stack(prototypes).to(DEVICE)

# Get combined dataset and class indices
@st.cache_data
def get_combined_dataset():
    full_dataset = datasets.ImageFolder(BASE_DATASET, transform)
    rare_dataset = datasets.ImageFolder(RARE_DATASET, transform)
    rare_dataset.samples = [(path, label + 5) for path, label in rare_dataset.samples]
    combined_dataset = ConcatDataset([full_dataset, rare_dataset])
    indices = {}
    for idx, (_, label) in enumerate(combined_dataset):
        indices.setdefault(label, []).append(idx)
    return combined_dataset, indices

# Visualize Prototypes Function
# Visualize Prototypes Function
def visualize_prototypes(model, dataset):
    # Step 1: Check if prototypes are available in session state
    prototypes = st.session_state.get("prototypes", None)

    if prototypes is None or prototypes.size(0) == 0:
        st.warning("⚠️ No prototypes found. Please train the model with few-shot learning before visualizing prototypes.")
        return

    # Step 2: Reduce the dimensionality using PCA (or t-SNE for better accuracy)
    pca = PCA(n_components=2)
    prototypes_2d = pca.fit_transform(prototypes.cpu().detach().numpy())

    # Step 3: Create a scatter plot
    plt.figure(figsize=(8, 6))

    # Define class labels (based on the dataset classes)
    class_names = sorted(os.listdir(BASE_DATASET)) + sorted(os.listdir(RARE_DATASET))  # base and rare classes

    # Step 4: Plot each prototype
    for i, prototype in enumerate(prototypes_2d):
        plt.scatter(prototype[0], prototype[1], label=class_names[i], s=100)

    # Add labels and title
    plt.title("Prototypes Visualization (PCA)")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.legend(loc='best')
    plt.grid(True)

    # Step 5: Display the plot using Streamlit
    st.pyplot(plt)


# Main app
st.title("🌿 Coffee Leaf Disease Classifier + Few-Shot Learning")

# Reset Model Button
if st.button("Reset Model"):
    st.session_state.model = reset_model()  # Reset model with standard prediction (no prototypes)
    if "prototypes" in st.session_state:
        del st.session_state["prototypes"]  # Remove stored prototypes
    st.success("✅ Model has been reset to the default prediction mode (no prototypes).")

option = st.radio("Choose an option", ["Upload & Predict", "Add New Class (Few-Shot)", "Train Few-Shot Model", "Visualize Prototypes"])

# Reload the model without projections or prototypes if it's in prediction mode
model = st.session_state.get('model', None)
if model is None:
    model = load_model()

if option == "Upload & Predict":
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        input_tensor = transform(image).unsqueeze(0).to(DEVICE)
        
        try:
            with torch.no_grad():
                prototypes = st.session_state.get("prototypes")
                
                if prototypes is not None and prototypes.size(0) > 0:
                    # If prototypes are available, perform prototype-based prediction
                    model.eval()  # Ensure model is in evaluation mode
                    embedding = model(input_tensor)  # Get embedding for the uploaded image
                    distances = torch.cdist(embedding, prototypes)  # Compare with prototypes
                    
                    # Apply softmax to normalize the confidence (optional but improves scaling)
                    softmax_confidence = torch.softmax(-distances, dim=1)
                    pred = torch.argmin(distances).item()  # Find the closest prototype
                    confidence = softmax_confidence[0, pred].item()  # Normalize and use confidence
                    
                    # Combine class names from base and rare datasets
                    base_classes = sorted(os.listdir(BASE_DATASET))  # Ensure proper order
                    rare_classes = sorted(os.listdir(RARE_DATASET))  # Ensure proper order
                    class_names = base_classes + rare_classes  # Combine class names for prediction
                    
                    predicted_class_name = class_names[pred]  # Get class name based on prediction
                    
                else:
                    # Fallback to base model if no prototypes are found
                    base_model = models.efficientnet_b0(weights=None)
                    base_model.classifier[1] = nn.Linear(base_model.classifier[1].in_features, 5)
                    base_model.load_state_dict(torch.load("E:\Model save\Deep_learning_model\model\efficientnet_coffee (1).pth", map_location=DEVICE))
                    base_model.to(DEVICE)
                    base_model.eval()
                    outputs = base_model(input_tensor)
                    probs = torch.softmax(outputs, dim=1)
                    pred = torch.argmax(probs, dim=1).item()
                    confidence = probs[0][pred].item()
                    class_names = sorted(os.listdir(BASE_DATASET))  # Only base dataset for base model
                    predicted_class_name = class_names[pred]
                    
            # Display the image and prediction result
            st.image(image, caption="📷 image has been uploaded", use_container_width=True)
            st.write(f"🔍 Predicted Class: {predicted_class_name}")
            st.write(f"📊 Confidence Score: {confidence * 100:.2f}%")
        except RuntimeError as e:
            st.error(f"Prediction failed: {e}. Please make sure the model is trained and try again.")

elif option == "Add New Class (Few-Shot)":
    new_class = st.text_input("Enter new class name")
    uploaded_files = st.file_uploader("Upload 10 images for new class", accept_multiple_files=True, type=["jpg", "jpeg", "png"])
    if st.button("Add Class") and new_class and len(uploaded_files) == 10:
        new_class_dir = os.path.join(RARE_DATASET, new_class)
        os.makedirs(new_class_dir, exist_ok=True)
        for i, file in enumerate(uploaded_files):
            img = Image.open(file).convert("RGB")
            save_path = os.path.join(new_class_dir, f"{i}.jpg")
            img.save(save_path)
            st.write(f"Saved: {save_path}")
        st.success(f"✅ Added new class: {new_class}. Please train few-shot model.")

    with st.expander("❌ Delete a Rare Class"):
        to_delete = st.selectbox("Select a rare class to delete", os.listdir(RARE_DATASET))
        if st.button("Delete Selected Class"):
            shutil.rmtree(os.path.join(RARE_DATASET, to_delete))
            st.success(f"✅ Deleted rare class: {to_delete}")

elif option == "Train Few-Shot Model":
    if "prototypes" in st.session_state:
        del st.session_state["prototypes"]

    # Get the combined dataset
    dataset, class_indices = get_combined_dataset()
    total_classes = len(class_indices)

    with st.form("train_form"):
        epochs = st.number_input("Epochs", 1, 50, 10)
        episodes_per_epoch = st.number_input("Episodes per epoch", 1, 100, 5)
        n_shot = st.number_input("Shots (support images/class)", 1, 10, 5)
        n_query = st.number_input("Queries (query images/class)", 1, 10, 5)
        submitted = st.form_submit_button("Start Few-Shot Training")

    if submitted:
        # Ensure n_way is not greater than the number of available classes
        if 5 > total_classes:
            st.warning(f"⚠️ Selected N-way (5) exceeds the number of available classes ({total_classes}). Please choose a smaller N-way.")
        else:
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)  # Adding weight decay
            model.train()

            progress_bar = st.progress(0)
            for epoch in range(epochs):
                total_loss, total_correct, total_samples = 0, 0, 0
                for _ in range(episodes_per_epoch):
                    s_imgs, s_labels, q_imgs, q_labels = create_episode(dataset, class_indices, n_way=5, n_shot=n_shot, n_query=n_query)
                    s_emb, q_emb = model(s_imgs), model(q_imgs)
                    loss = proto_loss(s_emb, q_emb, 5, n_shot, n_query)
                    
                    # During training, calculate prototypes
                    prototypes = s_emb.view(5, n_shot, -1).mean(dim=1)  # Calculate prototypes
                    distances = torch.cdist(q_emb, prototypes)  # Calculate distances to prototypes
                    preds = torch.argmin(distances, dim=1)  # Get predictions

                    total_correct += (preds == q_labels).sum().item()
                    total_samples += q_labels.size(0)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                    
                st.write(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss/episodes_per_epoch:.4f} - Accuracy: {total_correct/total_samples*100:.2f}%")
                progress_bar.progress((epoch + 1) / epochs)
            st.session_state.prototypes = calculate_prototypes(model, dataset)
            st.success("✅ Training done! Prototypes updated.")
        
elif option == "Visualize Prototypes":
    st.write("Visualizing prototypes for each class...")
    # Get the combined dataset
    dataset, _ = get_combined_dataset()
    visualize_prototypes(model, dataset)