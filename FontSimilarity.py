import torch
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
import os
import Grouping
from PIL import Image

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

def font_sim(map_name_in_strec, i, j):
    '''
    :param map_name_in_strec: map from which crops are compared, used for filepath
    :param i: index of first crop, used for filepath
    :param j: index of second crop, used for filepath
    :return: float from 0.0 to 1.0
    '''

    crop1 = Image.open(f"extracted_crops/{map_name_in_strec}_" + str(i) + ".jpeg")
    crop2 = Image.open(f"extracted_crops/{map_name_in_strec}_" + str(j) + ".jpeg")

    crop1 = transform(crop1).unsqueeze(0)
    crop2 = transform(crop2).unsqueeze(0)

    crop1 = crop1.to(device)
    crop2 = crop2.to(device)

    output = model(crop1, crop2)
    output = output.squeeze()

    output = output.item()

    return output

def check_for_crop(df, i, map_name_in_strec):
    if os.path.exists(f"extracted_crops/{map_name_in_strec}_" + str(i) + ".jpeg"):
        pass
    else:
        img_tmp = Grouping.polygon_crop(df.iloc[i]['polygons'], Image.open("processed/strec/" + map_name_in_strec + "/raw.jpeg"))
        img_tmp.save(f"extracted_crops/{map_name_in_strec}_" + str(i) + ".jpeg")

def calc_font_similarities(df, map_name_in_strec):
    f_sims = []
    neighbours = df['neighbours']
    for i in range(len(df)):
        if len(neighbours[i]) > 0:
            check_for_crop(df, i, map_name_in_strec)
        curr_i_score_dict = {}
        for j in neighbours[i]:
            check_for_crop(df, j, map_name_in_strec)
            curr_i_score_dict[j] = font_sim(map_name_in_strec, i , j)
        f_sims.append(curr_i_score_dict)

    df['font_scores'] = f_sims
    return df

def get_similarity_metric(df, i, j):
    i_has_j = False
    j_has_i = False

    i_scores = df.loc[i]['font_scores']
    j_scores = df.loc[j]['font_scores']
    if j in i_scores.keys():
        i_has_j = True
    elif i in j_scores.keys():
        j_has_i = True

    if i_has_j == True and j_has_i == True:
        return max(i_scores[j], j_scores[i])
    elif i_has_j == True:
        return i_scores[j]
    elif j_has_i == True:
        return j_scores[i]
    else:
        return 0