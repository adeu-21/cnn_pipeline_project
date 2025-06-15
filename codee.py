import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import torch.nn.functional as F
import time

# Define the first stage of the CNN
class Stage0(nn.Module):
    def __init__(self):
        super(Stage0, self).__init__()
        # Convolutional and pooling layers
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

    def forward(self, x):
        # Apply convolution, activation, and pooling
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
        x = x.view(-1, 1024 * 2 * 2)  # Flatten for next stage
        return x

# Define the second stage of the CNN
class Stage1(nn.Module):
    def __init__(self):
        super(Stage1, self).__init__()
        # Fully connected layers
        self.fc1 = nn.Linear(1024 * 2 * 2, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 128)
        self.fc5 = nn.Linear(128, 64)
        self.fc6 = nn.Linear(64, 10)

    def forward(self, x):
        # Apply fully connected layers with activation
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = self.fc6(x)
        return x

# Instantiate the two stages and place them on separate GPUs
model_stage0 = Stage0().to("cuda:0")
model_stage1 = Stage1().to("cuda:1")

# Load and prepare CIFAR-10 dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
dataset = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)

# Split dataset into training and testing
train_size = int(0.9 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(list(model_stage0.parameters()) + list(model_stage1.parameters()), lr=0.001)

# Log results to a file
output_file = open("output_para.txt", "w")

# Train the model using manual pipeline parallelism
def train(train_loader, criterion, optimizer, epochs=20):
    total_start_time = time.time()
    for epoch in range(epochs):
        model_stage0.train()
        model_stage1.train()
        running_loss = 0.0

        for step, (images, labels) in enumerate(train_loader, 1):
            images = images.to("cuda:0")
            labels = labels.to("cuda:1")

            # Forward pass through both stages
            intermediate = model_stage0(images).to("cuda:1")
            outputs = model_stage1(intermediate)

            # Backward pass and optimization
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            for param in model_stage0.parameters():
                if param.grad is not None:
                    param.grad = param.grad.to("cuda:0")
            optimizer.step()

            running_loss += loss.item()

            # Write average loss for every 100 steps to output file
            if step % 100 == 0:
                avg_loss = running_loss / step
                output_file.write(f"Epoch [{epoch+1}/{epochs}], Step [{step}/{len(train_loader)}], Loss: {avg_loss:.4f}\n")

        epoch_end_time = time.time()
        output_file.write(f"Epoch {epoch+1}, Loss: {running_loss / len(train_loader):.4f}, Time: {epoch_end_time - total_start_time:.2f}s\n")

    output_file.write(f"Total training time: {time.time() - total_start_time:.2f}s\n")

# Test the model
def test(test_loader):
    model_stage0.eval()
    model_stage1.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to("cuda:0")
            labels = labels.to("cuda:1")
            outputs = model_stage1(model_stage0(images).to("cuda:1"))
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
