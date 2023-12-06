import os
import glob
from itertools import combinations
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import json

def get_images():
    images = glob.glob(os.path.join('dependencies', 'ground_truth_labels', 'cropped', '*.jpeg'))
    return images

json_dict = {}
json_list = []

images = get_images()

is_same = False
end_loop = False

def on_button_yes_click(event):
    global is_same
    is_same = True
    json_list.append([image1, image2, 1])
    plt.close()

def on_button_no_click(event):
    global is_same
    is_same = False
    json_list.append([image1, image2, 0])
    plt.close()

def on_button_cancel_click(event):
    global end_loop
    end_loop = True
    json_dict['data'] = json_list
    with open('dependencies/ground_truth_labels/ground_truth_labels.json', 'w') as f:
        json.dump(json_dict, f)
    plt.close()

starting_num = 1400

j = 0
for image1, image2 in combinations(images, 2):
    if starting_num == j:
        # Load json into list
        with open('dependencies/ground_truth_labels/ground_truth_labels.json', 'r') as f:
            json_dict = json.load(f)
            if 'data' in json_dict:
                json_list = json_dict['data'][:starting_num]
                with open('dependencies/ground_truth_labels/ground_truth_labels_backup.json', 'w') as f:
                    json.dump(json_dict, f)
    if starting_num > j:
        j += 1
        continue
    if end_loop:
        print("Annotated {} images".format(starting_num - 1))
        break
    is_same = False
    # Show two images in one plot
    img1 = plt.imread(image1)
    img2 = plt.imread(image2)
    fig, ax = plt.subplots(1, 2)
    plt.subplots_adjust(bottom=0.2)
    ax[0].imshow(img1)
    ax[1].imshow(img2)
    button_cancel_ax = plt.axes([0.7, 0.05, 0.1, 0.075])
    button_no_ax = plt.axes([0.5, 0.05, 0.1, 0.075])
    button_yes_ax = plt.axes([0.3, 0.05, 0.1, 0.075])
    button_skip_ax = plt.axes([0.1, 0.05, 0.1, 0.075])
    button_yes = Button(button_yes_ax, 'Yes')
    button_yes.on_clicked(on_button_yes_click)
    button_no = Button(button_no_ax, 'No')
    button_no.on_clicked(on_button_no_click)
    button_cancel = Button(button_cancel_ax, 'Cancel')
    button_cancel.on_clicked(on_button_cancel_click)
    button_skip = Button(button_skip_ax, 'Skip')
    button_skip.on_clicked(lambda x: plt.close())

    plt.show()
    starting_num += 1
    j += 2
