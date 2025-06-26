import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from src.models.train_model import PneumoniaCNN
from src.data.load_data import load_data

def predict_batch():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model
    model = PneumoniaCNN().to(device)
    model.load_state_dict(torch.load("pneumonia_cnn.pt", map_location=device))
    model.eval()

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

# Only call this if running predict_batch directly
if __name__ == "__main__":
    predict_batch()
    show_misclassified(misclassified)
