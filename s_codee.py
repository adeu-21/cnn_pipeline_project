import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import torch.nn.functional as F
import time

# Define the CNN model
class FullModel(nn.Module):
    def __init__(self):
        super(FullModel, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv4 = nn.Conv2d(128, 256, 3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        self.conv5 = nn.Conv2d(256, 512, 3, padding=1)
        self.conv6 = nn.Conv2d(512, 512, 3, padding=1)
        self.pool3 = nn.MaxPool2d(2, 2)
        
        self.conv7 = nn.Conv2d(512, 1024, 3, padding=1)
        self.conv8 = nn.Conv2d(1024, 1024, 3, padding=1)
        self.pool4 = nn.MaxPool2d(2, 2)
        
        # Fully connected layers
        self.fc1 = nn.Linear(1024 * 2 * 2, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 128)
        self.fc5 = nn.Linear(128, 64)
        self.fc6 = nn.Linear(64, 10)

    def forward(self, x):
        # Apply convolutional layers
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool1(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool2(x)
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = self.pool3(x)
        x = F.relu(self.conv7(x))
        x = F.relu(self.conv8(x))
        x = self.pool4(x)
        
        # Flatten and apply fully connected layers
        x = x.view(-1, 1024 * 2 * 2)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = self.fc6(x)
        
        return x

# Create the model and move it to GPU
model = FullModel().to("cuda:0")

# Prepare CIFAR-10 dataset
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)
dataset = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)

# Split dataset into training and testing sets
train_size = int(0.9 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Set up loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Open a file to log results
output_file = open("output_nonpara.txt", "w")

# Train the model
def train(train_loader, criterion, optimizer, epochs=20):
    total_start_time = time.time()
    for epoch in range(epochs):
        epoch_start_time = time.time()
        model.train()
        running_loss = 0.0

        for step, (images, labels) in enumerate(train_loader, 1):
            images, labels = images.to("cuda:0"), labels.to("cuda:0")

            # Forward and backward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # Log average loss every 100 steps
            if step % 100 == 0:
                avg_loss = running_loss / step
                output_file.write(
                    f"Epoch [{epoch+1}/{epochs}], Step [{step}/{len(train_loader)}], Average Loss: {avg_loss:.4f}\n"
                )

        # Log final loss for the epoch
        epoch_end_time = time.time()
        output_file.write(
            f"Epoch {epoch+1}, Final Average Loss: {running_loss / len(train_loader):.4f}, "
            f"Time: {epoch_end_time - epoch_start_time:.2f} seconds\n"
        )

    # Log total training time
    total_end_time = time.time()
    output_file.write(f"Total training time: {total_end_time - total_start_time:.2f} seconds\n")

# Test the model
def test(test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to("cuda:0"), labels.to("cuda:0")

            # Forward pass and calculate accuracy
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    output_file.write(f"Test Accuracy: {accuracy:.2f}%\n")

# Train and test the model
train(train_loader, criterion, optimizer, epochs=20)
test(test_loader)

# Close the log file
output_file.close()
