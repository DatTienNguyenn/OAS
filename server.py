from flask import Flask, render_template, request
from torchvision import models, transforms
from PIL import Image
import torch
import os
import numpy as np
import pre as pr

# Khởi tạo Flask
app = Flask(__name__)

# Thiết lập device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Định nghĩa các lớp (classes)
classes = ['0', '1', '2', '3', '4']
class_names = {0: '0', 1: '1', 2: '2', 3: '3', 4: '4'}

# Hàm tiền xử lý ảnh (lấy từ notebook)
def default_preprocess(image):
    # Step 1: Crop to square
    w, h = image.size
    if w != h:
        crop_size = min(w, h)
        left = (w - crop_size) // 2
        top = (h - crop_size) // 2
        image = transforms.functional.crop(image, top, left, crop_size, crop_size)

    # Step 2: Resample to 299x299 (EfficientNet-B7 input size from notebook)
    image = transforms.functional.resize(image, (299, 299), interpolation=Image.BICUBIC)

    # Step 3 & 4: Center and scale pixel values
    image = transforms.functional.to_tensor(image)
    mean = image.mean()
    std = image.std() if image.std() > 0 else 1.0
    image = (image - mean) / std

    # Step 5: Rescale to [0, 1]
    image = (image - image.min()) / (image.max() - image.min())

    # Step 6: Replicate to 3 channels if grayscale
    if image.shape[0] == 1:
        image = image.repeat(3, 1, 1)

    # Step 7 & 8: Normalize with ImageNet stats
    image = transforms.functional.normalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    return image.unsqueeze(0)  # Thêm batch dimension

# Hàm tạo và tải mô hình
def load_model(model_path):
    model = models.efficientnet_b7(pretrained=False)
    num_features = model.classifier[1].in_features
    model.classifier[1] = torch.nn.Linear(num_features, len(classes))
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model.to(device)

# Tải mô hình
model_path = os.path.join("models", "best_efficientB7.pth")
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file {model_path} not found. Please ensure it is in the 'models' directory.")

model = load_model(model_path)
print(f"Loaded model from {model_path}")

@app.route("/", methods=['GET', 'POST'])
def home_page():
    if request.method == "POST":
        try:
            # Lấy file gửi lên
            image = request.files['file']
            if image:
                # Lưu file
                path_to_save = os.path.join("test/files", image.filename)
                os.makedirs(os.path.dirname(path_to_save), exist_ok=True)  # Tạo thư mục nếu chưa tồn tại
                print("Save = ", path_to_save)
                image.save(path_to_save)

                # Đọc và tiền xử lý ảnh
                img = Image.open(path_to_save).convert('RGB')
                input_tensor = default_preprocess(img).to(device)

                # Dự đoán
                with torch.no_grad():
                    output = model(input_tensor)
                    probabilities = torch.softmax(output, dim=1)
                    predicted_class = torch.argmax(probabilities, dim=1).item()
                    predicted_prob = probabilities[0][predicted_class].item()

                # Lấy top 3 dự đoán
                probs, indices = torch.topk(probabilities, 3)
                top_probs = probs[0].cpu().numpy()
                top_indices = indices[0].cpu().numpy()
                more = "{}: {:.2f}%; {}: {:.2f}%; {}: {:.2f}%".format(
                    class_names[top_indices[0]], top_probs[0] * 100,
                    class_names[top_indices[1]], top_probs[1] * 100,
                    class_names[top_indices[2]], top_probs[2] * 100
                )
                predict_name = class_names[predicted_class]

                return render_template("index.html",
                                      image=image.filename,
                                      msg="Upload successful!",
                                      predict_name=predict_name,
                                      more=more)
            else:
                return render_template('index.html', msg='Choose file to upload!')
        except Exception as ex:
            print(f"Error: {ex}")
            return render_template('index.html', msg="Can not analyze the image!")
    else:
        return render_template('index.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=False)