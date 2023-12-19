from sklearn.cluster import DBSCAN
import numpy as np
import json
from PIL import Image, ImageDraw
import random


def get_feature(x_points, y_points):

    bounding_box_corner = [min(x_points), min(y_points), max(x_points), max(y_points)]

    center = [np.mean(x_points), np.mean(y_points)]

    #directions = [np.rad2deg(np.arctan2(y_points[i] - y_points[i-1], x_points[i] - x_points[i-1])) for i in range(1, len(x_points))]

    #dir_hist = np.histogram(directions, bins=8, range=(-180, 180))[0]

    return bounding_box_corner + center #+ list(dir_hist)


def random_color():

    return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))


def visualize_polygons(clustered, img_path):

    # load the image
    image = Image.open(img_path)
    draw = ImageDraw.Draw(image)

    # choose a color for each cluster
    label_to_color = {str(label): random_color() for label in range(len(clustered))}

    # draw polygons
    for label, cluster in clustered.items():
        if label == "-1":
            continue
        else:
            color = label_to_color[label]
        pil_color = tuple(color)
        for polygon in cluster:
            draw.polygon(list(zip(polygon['polygon_x'], polygon['polygon_y'])), fill=pil_color, outline=pil_color)

    return image


def cluster_polygons(p_labels, conf_threshold = 0):

    features = []
    for i in range(len(p_labels['polygon_x'])):
        #if p_labels['score'] > conf_threshold:
            f = get_feature(p_labels['polygon_x'][str(i)], p_labels['polygon_y'][str(i)])
            features.append(f)
    #features = np.array(features)
    #features -= np.mean(features, axis=0)
    #features /= np.std(features, axis=0)

    clustering = DBSCAN(eps=50, min_samples=3).fit(features)
    c_labels = clustering.labels_

    clustered = {}
    new_label = len(set(c_labels)) - 1
    for i, c_label in enumerate(c_labels):
        if c_label == -1:
            continue
            clustered[str(new_label)] = [{'polygon_x': p_labels['polygon_x'][str(i)],
                                            'polygon_y': p_labels['polygon_y'][str(i)],
                                            'text': p_labels['text'][str(i)],
                                            'score': p_labels['score'][str(i)]}]
            new_label += 1
        else:
            if str(c_label) not in clustered.keys():
                clustered[str(c_label)] = [{'polygon_x': p_labels['polygon_x'][str(i)],
                                            'polygon_y': p_labels['polygon_y'][str(i)],
                                            'text': p_labels['text'][str(i)],
                                            'score': p_labels['score'][str(i)]}]
            else:
                clustered[str(c_label)].append({'polygon_x': p_labels['polygon_x'][str(i)],
                                                'polygon_y': p_labels['polygon_y'][str(i)],
                                                'text': p_labels['text'][str(i)],
                                                'score': p_labels['score'][str(i)]})

    return clustered


#with open('test/combined_tagged_all_layers.json', 'r', encoding='utf-8') as f:

#    clustered = cluster_polygons(json.load(f), conf_threshold=.5)

#    # visualize clusters
#    image = visualize_polygons(clustered, 'test/combined.png')
#    image.save('test/clustering.png')

#    with open('test/combined_tagged_all_layers_clustered.json', 'w', encoding='utf-8') as file:
#        json.dump(clustered, file, ensure_ascii=False)
