import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from src.data.load_data import load_data

# ----- Define a Simple CNN -----
class PneumoniaCNN(nn.Module):
    def __init__(self):
        super(PneumoniaCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),  # 1 channel grayscale
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 2)  # 2 classes: pneumonia, normal
        )

    def forward(self, x):
        return self.fc_layers(self.conv_layers(x))

# ----- Training Loop -----
def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PneumoniaCNN().to(device)

    train_loader, val_loader, test_loader = load_data()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.8)

    for epoch in range(10):  # or more!
        model.train()
        running_loss = 0.0
        correct, total = 0, 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.squeeze().to(device)
        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    train_acc = 100 * correct / total

    # 🧪 Validation Loop
    model.eval()
    val_correct, val_total = 0, 0
    with torch.no_grad():
        for val_images, val_labels in val_loader:
            val_images, val_labels = val_images.to(device), val_labels.squeeze().to(device)
            val_outputs = model(val_images)
            _, val_predicted = torch.max(val_outputs, 1)
            val_total += val_labels.size(0)
            val_correct += (val_predicted == val_labels).sum().item()

    val_acc = 100 * val_correct / val_total

    print(f"Epoch {epoch+1} | Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}% | Loss: {running_loss:.4f}")
    scheduler.step()

    current_lr = scheduler.get_last_lr()[0]
    print(f"🔁 Learning Rate after Epoch {epoch+1}: {current_lr:.6f}")


if __name__ == "__main__":
    train()
