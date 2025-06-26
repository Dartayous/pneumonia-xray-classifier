import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from src.models.train_model import PneumoniaCNN
from src.data.load_data import load_data
from PIL import Image

activations = {}
gradients = {}

def forward_hook(module, input, output):
    activations["value"] = output.detach()

def backward_hook(module, grad_input, grad_output):
    gradients["value"] = grad_output[0].detach()


def predict_batch():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model
    model = PneumoniaCNN().to(device)
    model.load_state_dict(torch.load("pneumonia_cnn.pt", map_location=device))
    model.eval()

    target_layer = model.conv_layers[4]  # adjust if needed
    target_layer.register_forward_hook(forward_hook)
    target_layer.register_backward_hook(backward_hook)

    _, _, test_loader = load_data()

    true_labels = []
    predicted_labels = []
    misclassified = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.squeeze().to(device)

            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            for img, pred, true_label in zip(images, preds, labels):
                if pred != true_label:
                    misclassified.append((img.cpu(), pred.cpu().item(), true_label.cpu().item()))

            true_labels.extend(labels.cpu().numpy())
            predicted_labels.extend(preds.cpu().numpy())

    # Accuracy
    correct = sum([p == t for p, t in zip(predicted_labels, true_labels)])
    total = len(true_labels)
    accuracy = 100 * correct / total
    print(f"🧪 Test Accuracy: {accuracy:.2f}%")

    cm = confusion_matrix(true_labels, predicted_labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Pneumonia", "Normal"])
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.show()

    # Visualize Grad-CAM for one misclassified example
    img, pred, true = misclassified[0]
    class_idx = pred  # or true if you want to see where it "should have" focused

    cam, class_idx = generate_gradcam(img, model, class_idx, device)

    # Visualization
    base_img = img.squeeze().numpy()
    cam_img = np.uint8(255 * cam.numpy())
    cam_img = Image.fromarray(cam_img).resize((28, 28), resample=Image.BICUBIC)
    cam_img = np.array(cam_img)

    plt.imshow(base_img, cmap="gray")
    plt.imshow(cam_img, cmap="jet", alpha=0.5)
    plt.title(f"Grad-CAM — Pred: {pred}, True: {true}")
    plt.axis("off")
    plt.show()


def show_misclassified(misclassified, classes=["Normal", "Pneumonia"], max_images=12):
    plt.figure(figsize=(12, 8))
    for idx, (img, pred, true) in enumerate(misclassified[:max_images]):
        plt.subplot(3, 4, idx + 1)
        img = img.squeeze().numpy()  # from 1x28x28 to 28x28
        plt.imshow(img, cmap="gray")
        plt.title(f"Pred: {classes[pred]}\nTrue: {classes[true]}")
        plt.axis("off")
    plt.tight_layout()
    plt.show()

def generate_gradcam(img, model, class_idx, device):
    input_tensor = img.unsqueeze(0).to(device).requires_grad_()

    # Forward
    output = model(input_tensor)

    # Backward
    model.zero_grad()
    output[0, class_idx].backward()

    # Grad & Activation
    grad = gradients["value"][0]
    act = activations["value"][0]
    weights = grad.mean(dim=(1, 2))
    cam = torch.relu((weights[:, None, None] * act).sum(0))
    cam = cam / cam.max()

    return cam.cpu(), class_idx

# Only call this if running predict_batch directly
if __name__ == "__main__":
    predict_batch()
    show_misclassified(misclassified)
