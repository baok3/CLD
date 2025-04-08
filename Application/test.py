import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import datasets, models
from PIL import Image
import os
import torch.nn.functional as F
from pathlib import Path
import json # Using json to save/load class labels for the few-shot model

# --- Configuration ---
try:
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if DEVICE.type == "cuda":
        test_tensor = torch.zeros(1).to(DEVICE)
        test_tensor = test_tensor + 1
except Exception as e:
    st.warning(f"CUDA error detected, falling back to CPU: {e}")
    DEVICE = torch.device("cpu")

DATASET_PATH = Path("./data_optimize/train_fewshot")
MODEL_SAVE_DIR = Path("./model/")
ORIGINAL_MODEL_NAME = "efficientnet_coffee.pth"
FEWSHOT_MODEL_NAME = "efficientnet_fewshot.pth"
FEWSHOT_LABELS_NAME = "efficientnet_fewshot_labels.json"

os.makedirs(DATASET_PATH, exist_ok=True)
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

ORIGINAL_MODEL_PATH = MODEL_SAVE_DIR / ORIGINAL_MODEL_NAME
FEWSHOT_MODEL_PATH = MODEL_SAVE_DIR / FEWSHOT_MODEL_NAME
FEWSHOT_LABELS_PATH = MODEL_SAVE_DIR / FEWSHOT_LABELS_NAME

# --- Define Original Classes (IMPORTANT!) ---
# This MUST match the order the ORIGINAL_MODEL_NAME was trained with
ORIGINAL_CLASS_LABELS = [
    'cerscospora',
    'healthy',
    'leaf rust',
    'miner',
    'phoma'
]
ORIGINAL_CLASS_TO_IDX = {label: i for i, label in enumerate(ORIGINAL_CLASS_LABELS)}
ORIGINAL_NUM_CLASSES = len(ORIGINAL_CLASS_LABELS)

# --- Helper Functions ---
def get_current_classes_from_dataset(dataset_dir):
    """Gets class labels from the current subdirectories in the dataset path."""
    if not os.path.isdir(dataset_dir):
        return [], {}
    classes = sorted([d.name for d in os.scandir(dataset_dir) if d.is_dir()])
    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    return classes, class_to_idx

def load_model_weights(model_path, num_classes):
    """Loads EfficientNet-B0 structure and loads weights from model_path."""
    model = models.efficientnet_b0(weights=None) # Start with an untrained structure
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, num_classes) # Adjust final layer

    if os.path.exists(model_path):
        try:
            # Load the state dictionary, making sure it's on the correct device
            model.load_state_dict(torch.load(model_path, map_location=DEVICE))
            st.success(f"Successfully loaded model weights from {model_path}")
        except Exception as e:
            st.error(f"Error loading model weights from {model_path}: {e}")
            st.warning("Model structure loaded, but weights failed. Predictions will be random.")
            return None # Indicate failure
    else:
        st.error(f"Model file not found at {model_path}. Cannot load weights.")
        return None # Indicate failure

    return model.to(DEVICE).eval() # Set to evaluation mode

# --- Transforms ---
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# --- Initialize Session State ---
if 'initialized' not in st.session_state:
    st.session_state['initialized'] = False
    st.session_state['model'] = None
    st.session_state['current_model_path'] = None
    st.session_state['class_labels'] = []
    st.session_state['num_classes'] = 0
    st.session_state['class_to_idx'] = {}

# --- Load Model and Class Info Once ---
if not st.session_state['initialized']:
    model_loaded = False
    loaded_model_path_str = "None"

    # Prioritize Few-Shot Model
    if os.path.exists(FEWSHOT_MODEL_PATH) and os.path.exists(FEWSHOT_LABELS_PATH):
        try:
            with open(FEWSHOT_LABELS_PATH, 'r') as f:
                fewshot_class_labels = json.load(f)
            if isinstance(fewshot_class_labels, list) and len(fewshot_class_labels) > 0:
                st.session_state['class_labels'] = fewshot_class_labels
                st.session_state['num_classes'] = len(fewshot_class_labels)
                st.session_state['class_to_idx'] = {label: i for i, label in enumerate(fewshot_class_labels)}
                st.session_state['model'] = load_model_weights(FEWSHOT_MODEL_PATH, st.session_state['num_classes'])
                if st.session_state['model'] is not None:
                     st.session_state['current_model_path'] = str(FEWSHOT_MODEL_PATH)
                     loaded_model_path_str = str(FEWSHOT_MODEL_PATH)
                     model_loaded = True
                else:
                    st.error("Failed to load few-shot model weights, check logs.")
                    # Reset state if model loading failed
                    st.session_state['class_labels'] = []
                    st.session_state['num_classes'] = 0
                    st.session_state['class_to_idx'] = {}

            else:
                 st.warning(f"Few-shot label file '{FEWSHOT_LABELS_NAME}' is empty or invalid. Trying original model.")
        except Exception as e:
            st.error(f"Error loading few-shot labels from {FEWSHOT_LABELS_PATH}: {e}. Trying original model.")

    # Fallback to Original Model if Few-Shot failed or doesn't exist
    if not model_loaded and os.path.exists(ORIGINAL_MODEL_PATH):
        st.session_state['class_labels'] = ORIGINAL_CLASS_LABELS
        st.session_state['num_classes'] = ORIGINAL_NUM_CLASSES
        st.session_state['class_to_idx'] = ORIGINAL_CLASS_TO_IDX
        st.session_state['model'] = load_model_weights(ORIGINAL_MODEL_PATH, ORIGINAL_NUM_CLASSES)
        if st.session_state['model'] is not None:
            st.session_state['current_model_path'] = str(ORIGINAL_MODEL_PATH)
            loaded_model_path_str = str(ORIGINAL_MODEL_PATH)
            model_loaded = True
        else:
            st.error("Failed to load original model weights. Predictions unavailable.")
            # Reset state if model loading failed
            st.session_state['class_labels'] = []
            st.session_state['num_classes'] = 0
            st.session_state['class_to_idx'] = {}

    if not model_loaded:
        st.error("No suitable model file found (original or few-shot). Predictions are disabled.")
        st.warning(f"Looked for: {ORIGINAL_MODEL_PATH} and {FEWSHOT_MODEL_PATH}")

    st.session_state['initialized'] = True
    # Use st.rerun() cautiously, it can cause loops. Only rerun if state changed meaningfully.
    # st.rerun() # Let's avoid rerun for now unless strictly necessary

# --- Main App UI ---
st.title("🌿 Ứng Dụng Nhận Diện Bệnh Lá Cà Phê")
st.write("Chụp ảnh hoặc tải ảnh lên để dự đoán bệnh.")
st.write(f"Current Model: `{st.session_state.get('current_model_path', 'None')}`")
st.write(f"Known Classes ({st.session_state.get('num_classes', 0)}): `{', '.join(st.session_state.get('class_labels', []))}`")

# --- Prediction Section ---
uploaded_file = st.file_uploader("📸 Tải ảnh lên để phân loại", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    if st.session_state.get('model') is None:
        st.error("Model not loaded. Cannot perform prediction.")
    elif not st.session_state.get('class_labels'):
        st.error("No classes defined for the current model. Cannot perform prediction.")
    else:
        try:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="📷 Ảnh đã tải lên", use_container_width=True)

            img_tensor = transform(image).unsqueeze(0).to(DEVICE)

            model_to_predict = st.session_state['model']
            model_to_predict.eval() # Ensure eval mode

            with torch.no_grad():
                output = model_to_predict(img_tensor)
                probabilities = F.softmax(output, dim=1)
                confidence, predicted_idx = torch.max(probabilities, 1)
                predicted_class_idx = predicted_idx.item()
                confidence_score = confidence.item() * 100

            if 0 <= predicted_class_idx < len(st.session_state['class_labels']):
                predicted_label = st.session_state['class_labels'][predicted_class_idx]
                st.write(f"🔍 Kết quả dự đoán: **{predicted_label}**")
                st.write(f"📊 Confidence Score: **{confidence_score:.2f}%**")
            else:
                st.error(f"Prediction index ({predicted_class_idx}) is out of range for the loaded model's known labels ({len(st.session_state['class_labels'])}). This indicates a mismatch between the model and its class list.")

        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")
            st.exception(e) # Show traceback for debugging


# --- Simplified Few-Shot Learning Section ---
with st.expander("🎯 Few-Shot Learning - Thêm Lớp Mới"):
    MIN_IMAGES_NEW_CLASS = 5

    uploaded_files_fs = st.file_uploader("Tải lên ít nhất 5 ảnh cho lớp mới",
                                       type=["png", "jpg", "jpeg"],
                                       accept_multiple_files=True,
                                       key="fewshot_uploader") # Unique key

    if uploaded_files_fs and len(uploaded_files_fs) >= MIN_IMAGES_NEW_CLASS:
        st.success(f"Đã tải lên {len(uploaded_files_fs)} ảnh (tối thiểu {MIN_IMAGES_NEW_CLASS})")

        new_class_name_input = st.text_input("Nhập tên lớp bệnh mới", key="new_class_name_input").strip() # Unique key

        if new_class_name_input:
            # Check against current dataset folders AND original labels to prevent conflicts
            current_folders, _ = get_current_classes_from_dataset(DATASET_PATH)
            if new_class_name_input in current_folders:
                st.error(f"Một thư mục tên '{new_class_name_input}' đã tồn tại trong {DATASET_PATH}. Vui lòng dọn dẹp hoặc chọn tên khác.")
            elif new_class_name_input in ORIGINAL_CLASS_LABELS and str(st.session_state.get('current_model_path')) == str(ORIGINAL_MODEL_PATH):
                 st.error(f"Tên lớp '{new_class_name_input}' trùng với một lớp gốc. Chọn tên khác.")
            else:
                new_class_folder = DATASET_PATH / new_class_name_input
                try:
                    os.makedirs(new_class_folder, exist_ok=True)
                    saved_count = 0
                    for i, file in enumerate(uploaded_files_fs):
                        try:
                            img_path = new_class_folder / f"image_{i+1}.jpg"
                            with open(img_path, "wb") as f:
                                f.write(file.getbuffer())
                            saved_count += 1
                        except Exception as e:
                            st.warning(f"Không thể lưu ảnh {file.name}: {e}")
                    st.success(f"Đã lưu {saved_count} ảnh vào thư mục '{new_class_folder}'")

                    if st.button("🚀 Cập nhật Model với Lớp Mới"):
                        if not st.session_state.get('model'):
                             st.error("Model gốc chưa được tải. Không thể thực hiện few-shot.")
                        else:
                            with st.spinner("🔄 Đang cập nhật model..."):
                                try:
                                    # 1. Get updated class list from folders
                                    updated_class_labels, updated_class_to_idx = get_current_classes_from_dataset(DATASET_PATH)
                                    updated_num_classes = len(updated_class_labels)

                                    if updated_num_classes <= st.session_state.get('num_classes', 0):
                                        st.error("Không tìm thấy thư mục lớp mới hoặc số lớp không tăng. Đảm bảo ảnh đã được lưu đúng cách.")
                                    else:
                                        st.info(f"Phát hiện {updated_num_classes} lớp trong thư mục dataset: {updated_class_labels}")

                                        # 2. Load the CURRENT model (could be original or previous few-shot)
                                        current_model = st.session_state['model']
                                        current_model.train() # Set to train mode for fine-tuning

                                        # 3. Freeze all layers except the classifier
                                        for param in current_model.parameters():
                                            param.requires_grad = False
                                        # Unfreeze ONLY the classifier's parameters
                                        for param in current_model.classifier.parameters():
                                             param.requires_grad = True

                                        # 4. Replace the classifier head
                                        num_ftrs = current_model.classifier[1].in_features
                                        current_model.classifier[1] = nn.Linear(num_ftrs, updated_num_classes)
                                        current_model = current_model.to(DEVICE) # Ensure the new layer is on the correct device

                                        # 5. Prepare dataset and dataloader ONLY with new data for fine-tuning the head
                                        # (Alternative: use full dataset if you want to tune more layers later)
                                        train_dataset = datasets.ImageFolder(root=DATASET_PATH, transform=transform)
                                        # Check if dataset loaded correctly
                                        if not train_dataset.classes:
                                             raise ValueError("ImageFolder không tìm thấy lớp nào. Kiểm tra lại cấu trúc thư mục.")
                                        if train_dataset.class_to_idx != updated_class_to_idx:
                                            st.warning(f"Dataset class mapping discrepancy: {train_dataset.class_to_idx} vs {updated_class_to_idx}. Using dataset's.")
                                            updated_class_labels = train_dataset.classes
                                            updated_class_to_idx = train_dataset.class_to_idx
                                            updated_num_classes = len(updated_class_labels)
                                            # Re-check and adjust classifier if needed (should be rare if folders were correct)
                                            if current_model.classifier[1].out_features != updated_num_classes:
                                                 current_model.classifier[1] = nn.Linear(num_ftrs, updated_num_classes).to(DEVICE)


                                        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=True)

                                        # 6. Define Loss and Optimizer for the NEW classifier head ONLY
                                        criterion = nn.CrossEntropyLoss()
                                        # Only optimize the parameters that require gradients (the new classifier)
                                        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, current_model.parameters()), lr=0.001)

                                        # 7. Fine-tuning loop (adjust epochs as needed)
                                        num_epochs = 10
                                        progress_bar = st.progress(0)
                                        status_text = st.empty()

                                        for epoch in range(num_epochs):
                                            running_loss = 0.0
                                            correct_preds = 0
                                            total_preds = 0
                                            for inputs, labels in train_loader:
                                                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

                                                optimizer.zero_grad()
                                                outputs = current_model(inputs)

                                                # Debugging: Check output shape and label range
                                                # st.write(f"Epoch {epoch}, Batch: Output shape {outputs.shape}, Labels min/max: {labels.min()}/{labels.max()}, Num Classes: {updated_num_classes}")
                                                if labels.min() < 0 or labels.max() >= updated_num_classes:
                                                    st.error(f"Label out of bounds detected! Label {labels.max()} vs Num Classes {updated_num_classes}. Check dataset folders/labels.")
                                                    raise IndexError("Label index out of range for model output")

                                                loss = criterion(outputs, labels)
                                                loss.backward()
                                                optimizer.step()

                                                running_loss += loss.item() * inputs.size(0)
                                                _, predicted = torch.max(outputs.data, 1)
                                                total_preds += labels.size(0)
                                                correct_preds += (predicted == labels).sum().item()

                                            epoch_loss = running_loss / len(train_loader.dataset)
                                            epoch_acc = (correct_preds / total_preds) * 100 if total_preds > 0 else 0
                                            status_text.text(f"Epoch {epoch+1}/{num_epochs} - Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%")
                                            progress_bar.progress((epoch + 1) / num_epochs)

                                        # 8. Save the fine-tuned model AND its class labels
                                        current_model.eval() # Set back to evaluation mode
                                        torch.save(current_model.state_dict(), FEWSHOT_MODEL_PATH)
                                        with open(FEWSHOT_LABELS_PATH, 'w') as f:
                                            json.dump(updated_class_labels, f)

                                        # 9. Update session state
                                        st.session_state['model'] = current_model
                                        st.session_state['current_model_path'] = str(FEWSHOT_MODEL_PATH)
                                        st.session_state['class_labels'] = updated_class_labels
                                        st.session_state['num_classes'] = updated_num_classes
                                        st.session_state['class_to_idx'] = updated_class_to_idx

                                        st.success(f"✅ Model đã được cập nhật thành công với {updated_num_classes} lớp!")
                                        st.info("App state updated. You can now predict with the new model.")
                                        # Clear the uploader and text input by rerunning part of the script
                                        st.experimental_rerun() # Use Streamlit's rerun mechanism


                                except FileNotFoundError as e:
                                     st.error(f"Dataset folder error during training: {e}. Ensure '{DATASET_PATH}' exists and has class subdirectories.")
                                except IndexError as e:
                                     st.error(f"Label index error during training: {e}. This often means the number of classes the model expects doesn't match the dataset labels. Check folder structure and class count.")
                                except Exception as e:
                                    st.error(f"Lỗi trong quá trình fine-tuning: {e}")
                                    st.exception(e)

                except Exception as e:
                    st.error(f"Lỗi khi tạo thư mục hoặc lưu ảnh: {e}")

    elif uploaded_files_fs:
         st.warning(f"Vui lòng tải lên ít nhất {MIN_IMAGES_NEW_CLASS} ảnh.")


# --- Option to Reset ---
st.sidebar.title("Options")
# Show reset button only if a few-shot model is currently loaded AND the original exists
if os.path.exists(ORIGINAL_MODEL_PATH) and st.session_state.get('current_model_path') == str(FEWSHOT_MODEL_PATH):
    if st.sidebar.button("🔄 Reset về Model Gốc"):
        with st.spinner("Đang reset về model gốc..."):
            try:
                # Load original model weights
                original_model = load_model_weights(ORIGINAL_MODEL_PATH, ORIGINAL_NUM_CLASSES)

                if original_model:
                     # Update session state
                    st.session_state['model'] = original_model
                    st.session_state['current_model_path'] = str(ORIGINAL_MODEL_PATH)
                    st.session_state['class_labels'] = ORIGINAL_CLASS_LABELS
                    st.session_state['num_classes'] = ORIGINAL_NUM_CLASSES
                    st.session_state['class_to_idx'] = ORIGINAL_CLASS_TO_IDX

                    # Attempt to remove few-shot files
                    try:
                        if os.path.exists(FEWSHOT_MODEL_PATH):
                            os.remove(FEWSHOT_MODEL_PATH)
                        if os.path.exists(FEWSHOT_LABELS_PATH):
                            os.remove(FEWSHOT_LABELS_PATH)
                        st.success("Đã reset về model gốc và xóa file few-shot.")
                    except OSError as e:
                        st.warning(f"Model reset, nhưng không thể xóa file few-shot: {e}")

                    st.experimental_rerun()
                else:
                    st.error("Không thể tải model gốc trong quá trình reset.")

            except Exception as e:
                st.error(f"Lỗi khi reset về model gốc: {e}")
elif st.session_state.get('current_model_path') == str(FEWSHOT_MODEL_PATH):
     st.sidebar.warning("Model gốc không tìm thấy. Không thể reset.")