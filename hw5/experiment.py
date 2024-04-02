import torch
import torch.nn as nn
import time
from torchvision import datasets, transforms

# Make sure to use only 10% of the available MNIST data.
# Otherwise, experiment will take quite long (around 90 minutes).
import numpy as np

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

learning_rate = 0.1
batch_size = 64
epochs = 150

def create_randomized_dataset(train_data, label_ratio):
    subset_indices = np.random.choice(len(train_data), len(train_data), replace=False)
    num_random_labels = int(len(train_data) * label_ratio)
    random_labels = torch.randint(low=0, high=10, size=(num_random_labels,))
    labels = torch.cat((train_data.targets[:num_random_labels], random_labels), dim=0)
    subset = torch.utils.data.Subset(train_data, subset_indices)
    subset.targets = labels
    return subset

# Evaluate the accuracy of the model on the test dataset
def test_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total * 100
    return accuracy

# (Modified version of AlexNet)
class AlexNet(nn.Module):
    def __init__(self, num_class=10):
        super(AlexNet, self).__init__()

        self.conv_layer1 = nn.Sequential(
            nn.Conv2d(1, 96, kernel_size=4),
            nn.ReLU(inplace=True),
            nn.Conv2d(96, 96, kernel_size=3),
            nn.ReLU(inplace=True)
        )
        self.conv_layer2 = nn.Sequential(
            nn.Conv2d(96, 256, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        self.conv_layer3 = nn.Sequential(
            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )

        self.fc_layer1 = nn.Sequential(
            nn.Dropout(),
            nn.Linear(6400, 800),
            nn.ReLU(inplace=True),
            nn.Linear(800, 10)
        )

    def forward(self, x):
        output = self.conv_layer1(x)
        output = self.conv_layer2(output)
        output = self.conv_layer3(output)
        output = torch.flatten(output, 1)
        output = self.fc_layer1(output)
        return output

# Load the MNIST dataset
train_data = datasets.MNIST(root='./', train=True, download=True, transform=transforms.ToTensor())

# Define the ratios of randomized labels to experiment with
label_ratios = [0, 0.2, 0.4, 0.6, 0.8, 1.0]

test_data = datasets.MNIST(root='./', train=False, download=True, transform=transforms.ToTensor())

# Define a data loader for the test dataset
test_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False)


device = torch.device("mps" if torch.cuda.is_available() else "cpu")
model = AlexNet().to(device)
loss_function = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
losses = []
times = []
acc = []

for label_ratio in label_ratios:
    print(f"\nExperiment with {int(label_ratio * 100)}% randomized labels")
    tmp_losses = []

    # Create a dataset subset with the specified label ratio
    train_subset = create_randomized_dataset(train_data, label_ratio)

    # Define a data loader for the subset
    batch_size = 64
    train_loader = torch.utils.data.DataLoader(dataset=train_subset, batch_size=batch_size, shuffle=True)

    # Define the model, optimizer, and loss function
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AlexNet().to(device)
    loss_function = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    # Train the model
    tick = time.time()
    for epoch in range(epochs):
        tmp2_losses = []
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            loss = loss_function(model(images), labels)
            tmp2_losses.append(loss.item())
            loss.backward()
            optimizer.step()
        tmp_losses.append(np.mean(tmp2_losses))
    tock = time.time()
    accuracy = test_model(model, test_loader)
    
    # logging
    losses.append(tmp_losses)
    times.append(tock-tick)
    acc.append(accuracy)
    
    print(f"Total training time: {tock - tick}")
