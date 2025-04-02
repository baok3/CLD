import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import datasets, models
from PIL import Image
import os
import random
import torch.nn.functional as F  # For softmax

# Định nghĩa thiết bị
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Đường dẫn dataset & mô hình
DATASET_PATH = "E:\\data_optimize\\train_fewshot"
MODEL_PATH = "E:/Model save/Deep_learning_model/efficientnet_coffee.pth"

# Load dataset để lấy danh sách class
train_dataset = datasets.ImageFolder(root=DATASET_PATH)
class_labels = train_dataset.classes

# Hàm load mô hình với đường dẫn đã chỉ định
@st.cache_resource
def load_model():
    model = models.efficientnet_b0(pretrained=False)  # Không tải pretrained weights
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, len(class_labels))  # Thay đổi số lớp đầu ra của classifier
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))  # Load trọng số mô hình đã huấn luyện
    return model.to(DEVICE).eval()  # Chuyển mô hình sang chế độ eval (dự đoán)

# Load mô hình ban đầu (mô hình đã huấn luyện trước)
model = load_model()

# Chuẩn hóa ảnh
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Dùng chuẩn hóa ImageNet
])

# **Giao diện App Chính**
st.title("🌿 Ứng Dụng Nhận Diện Bệnh Lá Cà Phê")
st.write("Chụp ảnh hoặc tải ảnh lên để dự đoán bệnh.")

# **Tải ảnh lên**
uploaded_file = st.file_uploader("📸 Tải ảnh lên để phân loại", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="📷 Ảnh đã tải lên", use_container_width=True)

    # Tiền xử lý ảnh
    img_tensor = transform(image).unsqueeze(0).to(DEVICE)

    # Dự đoán với mô hình
    with torch.no_grad():
        output = model(img_tensor)
        predicted_class = torch.argmax(output, dim=1).item()

        # Apply softmax to get the confidence scores
        probabilities = F.softmax(output, dim=1)
        confidence_score = probabilities[0][predicted_class].item() * 100  # Multiply by 100 to get percentage

    st.write(f"🔍 Kết quả dự đoán: **{class_labels[predicted_class]}**")
    st.write(f"📊 Confidence Score: **{confidence_score:.2f}%**")

# **Chức năng Few-Shot Learning**
with st.expander("🎯 Few-Shot Learning - Thêm Lớp Mới"):
    # Bước 1: Tải ảnh lên trước
    uploaded_files = st.file_uploader("📸 Tải lên 5 ảnh cho lớp mới", type=["png", "jpg", "jpeg"], accept_multiple_files=True)

    if uploaded_files and len(uploaded_files) == 5:
        st.success("✅ Đã tải lên đủ 5 ảnh! Hãy đặt tên cho lớp mới.")

        # Bước 2: Nhập tên lớp mới sau khi đã tải ảnh lên
        new_class_name = st.text_input("🆕 Nhập tên lớp bệnh mới và nhấn ENTER")

        if new_class_name:
            new_class_folder = os.path.join(DATASET_PATH, new_class_name)  # Lưu trực tiếp vào dataset gốc
            os.makedirs(new_class_folder, exist_ok=True)

            # Lưu ảnh vào thư mục mới
            for i, file in enumerate(uploaded_files):
                img_path = os.path.join(new_class_folder, f"image_{i+1}.jpg")
                with open(img_path, "wb") as f:
                    f.write(file.getbuffer())
            st.success(f"✅ Đã lưu 5 ảnh cho lớp {new_class_name}!")

            # Cập nhật danh sách lớp
            class_labels.append(new_class_name)

            # **Fine-tune mô hình với Few-Shot Learning**
            if st.button("🚀 Train Few-Shot Model"):
                st.info("🔄 Đang huấn luyện mô hình với lớp mới...")

                # Load lại dataset
                combined_dataset = datasets.ImageFolder(root=DATASET_PATH)
                class_indices = {cls: [i for i, (img, label) in enumerate(combined_dataset.samples) if label == cls] for cls in range(len(class_labels))}

                # Fine-tune mô hình
                def fine_tune_with_fewshot(model, dataset, class_indices, n_epochs=10):
                    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)  # Giảm learning rate để tránh overfitting
                    model.train()

                    for epoch in range(n_epochs):
                        total_loss = 0
                        for _ in range(5):  # 5 episodes mỗi epoch
                            # Tạo episode ngẫu nhiên từ các lớp có đủ ảnh hỗ trợ
                            sampled_classes = random.sample(list(class_indices.keys()), len(class_indices))

                            support_imgs, query_imgs, support_labels, query_labels = [], [], [], []

                            # Lọc ra các lớp có đủ ít nhất 3 ảnh hỗ trợ
                            sampled_classes = [cls for cls in sampled_classes if len(class_indices[cls]) >= 3]

                            for new_label, class_id in enumerate(sampled_classes):
                                indices = class_indices[class_id]
                                sampled_indices = random.sample(indices, min(10, len(indices)))

                                # Lấy 3 ảnh hỗ trợ và phần còn lại làm query
                                support_imgs += [dataset[i][0] for i in sampled_indices[:3]]
                                query_imgs += [dataset[i][0] for i in sampled_indices[3:]]
                                support_labels += [new_label] * 3
                                query_labels += [new_label] * (len(sampled_indices) - 3)

                            # Áp dụng transform và chuyển đổi sang tensor trước khi sử dụng torch.stack
                            support_imgs = torch.stack([transform(img).to(DEVICE) for img in support_imgs])
                            query_imgs = torch.stack([transform(img).to(DEVICE) for img in query_imgs])

                            # Tính embeddings
                            support_emb = model(support_imgs)
                            query_emb = model(query_imgs)

                            # Tính prototypes từ support_emb
                            prototypes = support_emb.view(len(sampled_classes), 3, -1).mean(dim=1)
                            distances = torch.cdist(query_emb, prototypes)

                            # Tạo labels cho ảnh query
                            labels = torch.tensor(query_labels).to(DEVICE)

                            # Tính loss
                            loss = nn.CrossEntropyLoss()(distances, labels)

                            optimizer.zero_grad()
                            loss.backward()
                            optimizer.step()
                            total_loss += loss.item()

                        st.write(f"🔄 Epoch [{epoch+1}/{n_epochs}], Loss: {total_loss/5:.4f}")

                fine_tune_with_fewshot(model, combined_dataset, class_indices)

                # Lưu lại mô hình sau khi fine-tuning
                torch.save(model.state_dict(), "E:/Model save/Deep_learning_model/efficientnet_fewshot.pth")
                st.success("✅ Mô hình đã học lớp mới thành công!")