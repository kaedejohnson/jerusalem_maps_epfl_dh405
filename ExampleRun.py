from ImageCrop import ImagePreprocessor
from PIL import Image
import os

import json
import pandas as pd
from SpotterWrapper import Spotter, PolygonVisualizer

img_path = r'raw_maps_20231024\1818_sieber\sieber_1818.png'

test_folder = r'test'

git_clone_location = 'C:/repo/'
spotter_directory = git_clone_location + 'mapkurator-spotter/spotter-v2'
model_weights = git_clone_location + 'detectron2-master/detectron2/checkpoint/model_v2_en.pth'
spotter_config = spotter_directory + '/configs/PALEJUN/Finetune/Rumsey_Polygon_Finetune.yaml'

image = Image.open(img_path)

image_preprocessor = ImagePreprocessor(image, overlapping_tolerance=0.2, num_layers=3, min_patch_resolution=256, max_patch_resolution=2048)

image_preprocessor.process()

print("preprocessing done")

spotter = Spotter(spotter_config, model_weights, confidence_thresh=0.8, draw_thresh=0.85)

all_layer_results = []

base_image_batch, base_offset_xs, base_offset_ys = image_preprocessor.get_image_patches(0)

vis = PolygonVisualizer()

vis.canvas_from_patches(base_image_batch, base_offset_xs, base_offset_ys)
#or
#vis.canvas_from_image(image)

for i in range(image_preprocessor.num_layers):
    # If you want to save patches for each layer, uncomment the following line
    #image_preprocessor.save_patches(os.path.join(test_folder, f'layer_{i}_patches'), layer=i)

    image_batch, offset_xs, offset_ys = image_preprocessor.get_image_patches(i)

    spotter.load_batch(image_batch, offset_xs, offset_ys)

    results = spotter.inference_batch()

    print(results)

    all_layer_results.extend(results)

    # If you want to save the tagged image for each layer, uncomment the following line
    #vis.draw(results).save(os.path.join(test_folder, f'combined_tagged_{i}.png'))

    # If you want to save polygons in json for each layer, uncomment the following line
    vis.save_json(results, os.path.join(test_folder, f'combined_tagged_{i}.json'))

# If you want to save the tagged image for all layers, uncomment the following line
vis.draw(all_layer_results).save(os.path.join(test_folder, f'combined_tagged_all_layers.png'))

# If you want to save polygons in json for all layers, uncomment the following line
vis.save_json(all_layer_results, os.path.join(test_folder, f'combined_tagged_all_layers.json'))