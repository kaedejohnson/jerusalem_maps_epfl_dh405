import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import json
from torch.utils.data import random_split
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
import numpy as np
import random
import cv2


with open("dependencies/ground_truth_labels/ground_truth_labels.json", "r", encoding="utf-8") as f:
    data_list = json.load(f)["data"]
    print(len(data_list))

# Define a custom dataset
class ImagePairDataset(Dataset):
    def __init__(self, data_list, transform=None):
        self.data_list = data_list
        self.transform = transform

    def __len__(self):
        return len(self.data_list)

    def pre_processing(self, img_path):
        # Load the image
        image = cv2.imread(img_path)
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        # Apply adaptive thresholding to get a binary image
        thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
        )
        # Apply morphological operations to clean up small dots/noise and close gaps in text
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        # Resize image to 128x128 as per your requirement
        resized = cv2.resize(morph, (128, 128), interpolation=cv2.INTER_AREA)
        # Convert to tensor
        tensor_image = transforms.ToTensor()(resized)

        # # If you need to view the processed image:
        # cv2.imshow("Processed Image", resized)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        return tensor_image

    def __getitem__(self, idx):
        img1_path, img2_path, label = self.data_list[idx]

        # img1 = Image.open(img1_path)
        # img2 = Image.open(img2_path)
        # if self.transform:
        #     img1 = self.transform(img1)
        #     img2 = self.transform(img2)

        img1 = self.pre_processing(img1_path)
        img2 = self.pre_processing(img2_path)

        return img1, img2, torch.tensor(label, dtype=torch.float32), img1_path, img2_path


# Define the CNN architecture
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.fc1 = nn.Linear(64 * 30 * 30 * 2, 128)  # Adjusted for concatenated features
        self.fc2 = nn.Linear(128, 1)

    def forward_one(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        return x

    def forward(self, x1, x2):
        x1 = self.forward_one(x1)
        x2 = self.forward_one(x2)
        x = torch.cat((x1, x2), 1)  # Concatenate the features
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return torch.sigmoid(x)


# Set a fixed seed for reproducibility
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multiGPU setups
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Instantiate the model
model = SimpleCNN()

# Loss function and optimizer
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Dataset and DataLoader
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])
dataset = ImagePairDataset(data_list, transform=transform)

# Split the dataset into training and testing sets
train_size = int(0.8 * len(dataset))  # 80% for training
test_size = len(dataset) - train_size  # 20% for testing
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# Create DataLoaders for training and testing sets
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# Check if CUDA is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# # Transfer the model to the GPU
# model.to(device)
#
# # Training loop
# num_epochs = 80
# for epoch in range(num_epochs):
#     model.train()  # Set the model to training mode
#     for img1, img2, labels in train_loader:
#         img1, img2, labels = img1.to(device), img2.to(device), labels.to(device)
#         outputs = model(img1, img2)
#         outputs = outputs.squeeze()
#         loss = criterion(outputs, labels)
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#     print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}')
#
#     if (epoch + 1) % 10 == 0:
#         # Evaluation with Confusion Matrix
#         model.eval()
#         all_labels = []
#         all_predictions = []
#
#         with torch.no_grad():
#             for img1, img2, labels, _, _ in test_loader:
#                 img1, img2, labels = img1.to(device), img2.to(device), labels.to(device)
#                 outputs = model(img1, img2)
#                 outputs = outputs.squeeze()
#                 predicted = (outputs > 0.5).float()
#
#                 all_labels.extend(labels.cpu().numpy())
#                 all_predictions.extend(predicted.cpu().numpy())
#
#         conf_matrix = confusion_matrix(all_labels, all_predictions)
#         accu = np.trace(conf_matrix) / np.sum(conf_matrix)
#         # Accuracy
#         print("Accuracy:", accu)
#         # Precision
#         print("Precision:", conf_matrix[1, 1] / (conf_matrix[1, 1] + conf_matrix[0, 1]))
#         # Recall
#         print("Recall:", conf_matrix[1, 1] / (conf_matrix[1, 1] + conf_matrix[1, 0]))
#         # Confusion Matrix
#         print("Confusion Matrix:\n", conf_matrix)
#         torch.save(model.state_dict(), f"font_disc_weights/FontDiscriminator_afterpad_{epoch + 1}_{accu * 100:.2f}.pth")


# evaluation
model.load_state_dict(torch.load("font_disc_weights/FontDiscriminator_afterpad_40_96.96.pth"))
model.to(device)
model.eval()
all_labels = []
all_predictions = []
wrong_pairs = []

with torch.no_grad():
    for img1, img2, labels, img1_path, img2_path in test_loader:
        img1, img2, labels = img1.to(device), img2.to(device), labels.to(device)
        outputs = model(img1, img2)
        outputs = outputs.squeeze()
        predicted = (outputs > 0.5).float()

        # print(img1_path[0])
        # print(img2_path[0])
        # print(labels.item())
        # print(predicted.item())

        all_labels.append(labels.item())
        all_predictions.append(predicted.item())

        if labels.item() != predicted.item():
            wrong_pairs.append([img1_path[0], img2_path[0], int(labels.item())])

        # all_labels.extend(labels.cpu().numpy())
        # all_predictions.extend(predicted.cpu().numpy())

conf_matrix = confusion_matrix(all_labels, all_predictions)
accu = np.trace(conf_matrix) / np.sum(conf_matrix)
# Accuracy
print("Accuracy:", accu)
# Precision
print("Precision:", conf_matrix[1, 1] / (conf_matrix[1, 1] + conf_matrix[0, 1]))
# Recall
print("Recall:", conf_matrix[1, 1] / (conf_matrix[1, 1] + conf_matrix[1, 0]))
# Confusion Matrix
print("Confusion Matrix:\n", conf_matrix)

with open('wrong_pairs.json', 'w') as file:
    json.dump({"data": wrong_pairs}, file)