import torch
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F


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


# Check if CUDA is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# Load the model
model = SimpleCNN()
model.load_state_dict(torch.load("font_disc_weights/FontDiscriminator_afterpad_40_96.96.pth"))
model.to(device)
model.eval()

# Data Transformation
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

def font_sim(crop1, crop2):
    '''
    :param crop1: PIL Image Object
    :param crop2: PIL Image Object
    :return: float from 0.0 to 1.0
    '''

    crop1 = transform(crop1).unsqueeze(0)
    crop2 = transform(crop2).unsqueeze(0)

    crop1 = crop1.to(device)
    crop2 = crop2.to(device)

    output = model(crop1, crop2)
    output = output.squeeze()

    output = output.item()

    return output