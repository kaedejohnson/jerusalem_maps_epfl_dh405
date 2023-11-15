from ImageCrop import ImagePreprocessor
from PIL import Image
import os

from SpotterWrapper import Spotter

img_path = r'raw_maps_20231024\1818_sieber\sieber_1818.png'

test_folder = r'test'

git_clone_location = 'C:/repo/'
spotter_directory = git_clone_location + 'mapkurator-spotter/spotter-v2'
model_weights = git_clone_location + 'detectron2-master/detectron2/checkpoint/model_v2_en.pth'
spotter_config = spotter_directory + '/configs/PALEJUN/Finetune/Base-SynthMap-Polygon.yaml'

image = Image.open(img_path)

image_preprocessor = ImagePreprocessor(image, overlapping_tolerance=0.2, num_layers=3, min_patch_resolution=256, max_patch_resolution=2048)

image_preprocessor.process()

print("preprocessing done")

spotter = Spotter(spotter_config, model_weights)

all_layer_instances = []
all_layer_offset_xs = []
all_layer_offset_ys = []

for i in range(image_preprocessor.num_layers):
    # If you want to save for each layer, uncomment the following line
    #image_preprocessor.save_patches(os.path.join(test_folder, f'layer_{i}_patches'), layer=i)

    image_batch, offset_xs, offset_ys = image_preprocessor.get_image_patches(i)

    spotter.load_batch(image_batch, offset_xs, offset_ys)

    spotter.inference_batch()

    all_layer_instances.extend(spotter.instances)
    all_layer_offset_xs.extend(offset_xs)
    all_layer_offset_ys.extend(offset_ys)

    # If you want to draw for each layer, uncomment the following line
    #spotter.draw(os.path.join(test_folder, f'combined_tagged_{i}.png'))

spotter.draw(os.path.join(test_folder, 'combined_tagged_all_layers.png'), draw_instances=all_layer_instances, draw_offset_xs=all_layer_offset_xs, draw_offset_ys=all_layer_offset_ys)
