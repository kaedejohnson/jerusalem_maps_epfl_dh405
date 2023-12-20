import random
import math
from shapely.geometry import Polygon, MultiPolygon
from collections import Counter
import json 
from itertools import combinations
import ExtractHandling
import SpotterWrapper
import numpy as np
import pandas as pd
import unidecode
import re
import scipy
from PIL import Image
import warnings
warnings.filterwarnings("ignore", message="Unpickling a shapely <2.0 geometry object.")


# Compared extracted labels to ground truth

def aggregate_components_to_multiline(multiline_g):

    longest_annotation = max(multiline_g['annotation'], key=len, default='')

    return pd.Series({
        'all_points_x': [point for sublist in multiline_g['all_points_x'] for point in sublist],
        'all_points_y': [point for sublist in multiline_g['all_points_y'] for point in sublist],
        'annotation': longest_annotation,
        'annotation_length': len(longest_annotation),
        'index': multiline_g.index[0]
    })

def load_ground_truth_labels(map_name_in_strec, multiline_handling, labels_on_fullsize_map=True):
    with open('dependencies/ground_truth_labels/' + map_name_in_strec + '.json', encoding='utf-8') as f:
        gt_labels_tmp = json.load(f)
        gt_labels = pd.DataFrame([
            {
                'all_points_x': obs['shape_attributes']['all_points_x'],
                'all_points_y': obs['shape_attributes']['all_points_y'],
                'annotation': obs['region_attributes']['annotation'],
                'multiline_g': obs['region_attributes'].get('multiline_g', None)
            }
            for obs in gt_labels_tmp[list(gt_labels_tmp.keys())[0]]['regions']
        ])

    gt_labels['annotation_length'] = gt_labels['annotation'].apply(len)
    tmp1 = gt_labels[~gt_labels['multiline_g'].apply(lambda x: pd.to_numeric(x, errors='coerce')).notnull()].copy()
    tmp1.loc[:, 'multiline_g'] = ""
    tmp2 = gt_labels[gt_labels['multiline_g'].apply(lambda x: pd.to_numeric(x, errors='coerce')).notnull()]

    if multiline_handling == 'largest':
        tmp2 = tmp2.groupby('multiline_g').apply(aggregate_components_to_multiline).reset_index().set_index('index')
        return pd.concat([tmp2, tmp1]).sort_index()
    elif multiline_handling == 'components':
        tmp2 = tmp2.drop(tmp2.groupby('multiline_g')['annotation_length'].idxmax()) # drop "wrap-around" label
        return pd.concat([tmp2, tmp1]).sort_index()

## Retain a subset of labels based on crop coordinates
def coords_fail_condition(list, direction_for_drop, value, baseline):
    if baseline == 1:
        return 1
    else:
        if direction_for_drop == '<':
            num_coords_broke_rule = sum([0 if coord < value else 1 for coord in list])
        elif direction_for_drop == '>':
            num_coords_broke_rule = sum([0 if coord > value else 1 for coord in list])
        if num_coords_broke_rule > 0:
            return 1
        else:
            return 0
        
def polygon_fail_condition(object, x_0_or_y_1, direction_for_drop, value, baseline):
    if baseline == 1:
        return 1
    else:
        if direction_for_drop == '<':
            if isinstance(object, Polygon):
                poly_broke_rule = any(coord[x_0_or_y_1] > value for coord in object.exterior.coords)
            elif isinstance(object, MultiPolygon):
                poly_broke_rule = any(any(coord[x_0_or_y_1] > value for coord in poly.exterior.coords) for poly in object.geoms)
        elif direction_for_drop == '>':
            if isinstance(object, Polygon):
                poly_broke_rule = any(coord[x_0_or_y_1] < value for coord in object.exterior.coords)
            elif isinstance(object, MultiPolygon):
                poly_broke_rule = any(any(coord[x_0_or_y_1] < value for coord in poly.exterior.coords) for poly in object.geoms)
        if poly_broke_rule == True:
            return 1
        else:
            return 0
        
def retain_crop_coords_only(df, left_x, right_x, top_y, bottom_y):
    df['drop'] = 0
    df['drop'] = df.apply(lambda row: coords_fail_condition(row['all_points_x'], '>', left_x, row['drop']), axis=1)
    df['drop'] = df.apply(lambda row: coords_fail_condition(row['all_points_x'], '<', right_x, row['drop']), axis=1)
    df['drop'] = df.apply(lambda row: coords_fail_condition(row['all_points_y'], '>', top_y, row['drop']), axis=1)
    df['drop'] = df.apply(lambda row: coords_fail_condition(row['all_points_y'], '<', bottom_y, row['drop']), axis=1)
    df = df[df['drop'] == 0]
    #print("retaining " + str(len(df)) + " labels fully inside crop area")
    return df

def retain_crop_polygons_only(df, left_x, right_x, top_y, bottom_y):
    df['drop'] = 0
    df['drop'] = df.apply(lambda row: polygon_fail_condition(row['label_polygons'], 0, '>', left_x, row['drop']), axis=1)
    df['drop'] = df.apply(lambda row: polygon_fail_condition(row['label_polygons'], 0, '<', right_x, row['drop']), axis=1)
    df['drop'] = df.apply(lambda row: polygon_fail_condition(row['label_polygons'], 1, '>', top_y, row['drop']), axis=1)
    df['drop'] = df.apply(lambda row: polygon_fail_condition(row['label_polygons'], 1, '<', bottom_y, row['drop']), axis=1)
    df = df[df['drop'] == 0]
    #print("retaining " + str(len(df)) + " labels fully inside crop area")
    return df

def remove_non_alphabetical_characters_and_accents(string):
    string = unidecode.unidecode(string.lower())
    string = re.sub(r'[^a-zA-Z ]+', '', string)
    return string

def retain_alphabetic_annotations_only(df):
    df = df.copy()
    df.loc[:, 'drop_txt'] = 0
    df = df.copy()
    df.loc[:, 'annotation_stripped'] = df['annotation'].apply(remove_non_alphabetical_characters_and_accents)
    annotation_cond_ind = df['annotation_stripped'].str.len() == 0
    df.loc[annotation_cond_ind, 'drop_txt'] = 1
    df = df[df['drop_txt'] == 0]
    #print("retaining " + str(len(df)) + " labels that have alphabetic characters")
    return df

## Calculate and Match IoUs
def calculate_IoU_matrix(spotter_labels, gt_labels):
    IoU_matrix = []
    for sptr_lab in spotter_labels:
        row = []
        for gt_lab in gt_labels:
            intersection_area = sptr_lab[0].intersection(gt_lab[0]).area
            union_area = sptr_lab[0].union(gt_lab[0]).area
            iou = intersection_area / union_area if union_area > 0 else 0
            row.append((iou, sptr_lab[1], gt_lab[1]))
        IoU_matrix.append(row)
    return np.array(IoU_matrix)

def maximize_1to1_IoU(IoU_matrix):
    IoUs_only = np.array([[float(col[0]) for col in row] for row in IoU_matrix])
    row_ind, col_ind = scipy.optimize.linear_sum_assignment(IoUs_only, maximize=True)
    IoU_pairs = IoU_matrix[row_ind, col_ind]
    return IoU_pairs

def geographic_evaluation(map_name_in_strec, multiline_handling, patches, methods = 'methods_1_2_3'):

    num_detected_tmp = []
    num_gt_tmp = []
    IoU_pairs_tmp = []

    for patch in patches:
        left_x, right_x, top_y, bottom_y = patch[0], patch[1], patch[2], patch[3]
        gt_labels_full = load_ground_truth_labels(map_name_in_strec, multiline_handling) # Evaluation.load_ground_truth_labels(map_name_in_strec, "components")
        gt_labels_crop = retain_crop_coords_only(gt_labels_full, left_x, right_x, top_y, bottom_y) 
        gt_labels_crop = retain_alphabetic_annotations_only(gt_labels_crop)
        gt_labels_crop = ExtractHandling.cast_coords_as_Polygons(gt_labels_crop) #ExtractHandling.cast_coords_as_Polygons(gt_labels_full)

        if methods == "methods_0":
            spotter_labels_full = ExtractHandling.load_spotter_labels(map_name_in_strec, "combined_tagged_1.json")
            spotter_labels_crop = retain_crop_coords_only(spotter_labels_full, left_x, right_x, top_y, bottom_y)
            if len(spotter_labels_crop) == 0:
                return 0, gt_labels_crop, np.array([['0.0', 'shell', 'array']])
            spotter_labels_crop = ExtractHandling.cast_coords_as_Polygons(spotter_labels_crop)
            spotter_labels_crop.rename(columns={'text': 'annotation'}, inplace=True)
        else:
            spotter_labels_full = ExtractHandling.load_processed_labels(map_name_in_strec, methods)
            spotter_labels_crop = retain_crop_polygons_only(spotter_labels_full, left_x, right_x, top_y, bottom_y)

        spotter_labels_crop = retain_alphabetic_annotations_only(spotter_labels_crop)

        IoU_matrix = calculate_IoU_matrix(list(spotter_labels_crop[['label_polygons', 'annotation']].itertuples(index=False, name=None)), list(gt_labels_crop[['label_polygons', 'annotation']].itertuples(index=False, name=None)))
        num_detected_tmp.append(IoU_matrix.shape[0])
        num_gt_tmp.append(IoU_matrix.shape[1])
        IoU_pairs_tmp.append(maximize_1to1_IoU(IoU_matrix))
    
    num_detected = sum(num_detected_tmp)
    num_gt = sum(num_gt_tmp)
    IoU_pairs = np.concatenate(IoU_pairs_tmp)

    return num_detected, num_gt, IoU_pairs

def edit_distance_similarity(word1, word2):
    m, n = len(word1), len(word2)
    if m == 0 or n == 0:
        return -1
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if word1[i - 1] == word2[j - 1]:
                cost = 0
            else:
                cost = 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)

    return 1 - pow(dp[m][n] / max(len(word1), len(word2)), 0.5)

def text_compare(s1, s2):
    s1_lwr_26 = remove_non_alphabetical_characters_and_accents(s1)
    s2_lwr_26 = remove_non_alphabetical_characters_and_accents(s2)
    return edit_distance_similarity(s1_lwr_26, s2_lwr_26)

def prec_rec(map_name_in_strec, multiline_handling, patches, methods = 'methods_1_2_3'):
    num_detected, num_gt, IoU_pairs = geographic_evaluation(map_name_in_strec, multiline_handling, patches, methods)

    IoU_pairs = pd.DataFrame(IoU_pairs, columns=['geo_IoU', 'spotter_txt', 'gt_txt'])
    IoU_pairs['normalized_txt_similarity'] = IoU_pairs.apply(lambda row: text_compare(row['spotter_txt'], row['gt_txt']), axis=1)


    geo_prec = IoU_pairs['geo_IoU'].astype(float).sum(axis=0) / num_detected
    text_prec = IoU_pairs['normalized_txt_similarity'].astype(float).sum(axis=0) / num_detected

    geo_rec = IoU_pairs['geo_IoU'].astype(float).sum(axis=0) / num_gt
    text_rec = IoU_pairs['normalized_txt_similarity'].astype(float).sum(axis=0) / num_gt

    print("Avg of Geographic Precision: " + str(geo_prec))
    print("Avg of Text Precision: " + str(text_prec))
    
    print("Avg of Geographic Recall: " + str(geo_rec))
    print("Avg of Text Recall: " + str(text_rec))

    return geo_prec, text_prec, geo_rec, text_rec, IoU_pairs, num_detected, num_gt


def plot_recovered_seq(map_name_in_strec, methods='methods_1_2_3'):

    # prepare polygons and texts to draw
    spotter_labels_full = ExtractHandling.load_processed_labels(map_name_in_strec, methods)
    polygons = spotter_labels_full['label_polygons'].tolist()
    texts = spotter_labels_full['annotation'].tolist()

    # drawing polygons
    vis = SpotterWrapper.PolygonVisualizer()
    canvas = Image.open(f'processed/strec/{map_name_in_strec}/raw.jpeg')
    vis.canvas_from_image(canvas)
    vis.draw_poly(polygons, texts, PCA_feature_list=None, BSplines=None, random_color=True)
    vis.save(f'processed/strec/{map_name_in_strec}/recovered_sequences.jpeg')
