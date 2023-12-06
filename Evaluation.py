import random
import math
from shapely.geometry import Polygon, MultiPolygon
from collections import Counter
import json 
from itertools import combinations
import ExtractHandling
import numpy as np
import pandas as pd
import unidecode
import re
import scipy

# Compared extracted labels to ground truth

def load_ground_truth_labels(map_name_in_strec, multiline_handling, labels_on_fullsize_map=True):
    with open('dependencies/ground_truth_labels/' + map_name_in_strec + '.json') as f:
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

    if multiline_handling == 'largest':
        gt_labels['annotation_length'] = gt_labels['annotation'].apply(len)
        tmp1 = gt_labels[gt_labels['multiline_g'].isnull()]
        tmp2 = gt_labels.dropna(subset=['multiline_g'])
        gt_labels = pd.concat([tmp2.loc[tmp2.groupby('multiline_g')['annotation_length'].idxmax()], tmp1])
    elif multiline_handling == 'components':
        gt_labels['annotation_length'] = gt_labels['annotation'].apply(len)
        tmp1 = gt_labels[gt_labels['multiline_g'].isnull()]
        tmp2 = gt_labels.dropna(subset=['multiline_g'])
        gt_labels = pd.concat([tmp2.loc[~tmp2.index.isin(tmp2.groupby('multiline_g')['annotation_length'].idxmax())], tmp1])
    return gt_labels

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
    print("retaining " + str(len(df)) + " labels fully inside crop area")
    return df

def retain_crop_polygons_only(df, left_x, right_x, top_y, bottom_y):
    df['drop'] = 0
    df['drop'] = df.apply(lambda row: polygon_fail_condition(row['label_polygons'], 0, '>', left_x, row['drop']), axis=1)
    df['drop'] = df.apply(lambda row: polygon_fail_condition(row['label_polygons'], 0, '<', right_x, row['drop']), axis=1)
    df['drop'] = df.apply(lambda row: polygon_fail_condition(row['label_polygons'], 1, '>', top_y, row['drop']), axis=1)
    df['drop'] = df.apply(lambda row: polygon_fail_condition(row['label_polygons'], 1, '<', bottom_y, row['drop']), axis=1)
    df = df[df['drop'] == 0]
    print("retaining " + str(len(df)) + " labels fully inside crop area")
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
    print("retaining " + str(len(df)) + " labels that have alphabetic characters")
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

def geographic_evaluation(map_name_in_strec, multiline_handling, coords, spotter_target = 'rectified'):
    left_x, right_x, top_y, bottom_y = coords[0], coords[1], coords[2], coords[3]
    gt_labels_full = load_ground_truth_labels(map_name_in_strec, multiline_handling) # Evaluation.load_ground_truth_labels(map_name_in_strec, "components")
    gt_labels_crop = retain_crop_coords_only(gt_labels_full, left_x, right_x, top_y, bottom_y) 
    gt_labels_crop = retain_alphabetic_annotations_only(gt_labels_crop)
    gt_labels_crop = ExtractHandling.cast_coords_as_Polygons(gt_labels_crop) #ExtractHandling.cast_coords_as_Polygons(gt_labels_full)

    if spotter_target == 'rectified':
        spotter_labels_full = ExtractHandling.load_rectified_polygons(map_name_in_strec)
        spotter_labels_crop = retain_crop_polygons_only(spotter_labels_full, left_x, right_x, top_y, bottom_y)
    else:
        spotter_labels_full = ExtractHandling.load_spotter_labels(map_name_in_strec, spotter_target)
        spotter_labels_crop = retain_crop_coords_only(spotter_labels_full, left_x, right_x, top_y, bottom_y)
        spotter_labels_crop = ExtractHandling.cast_coords_as_Polygons(spotter_labels_crop)
        spotter_labels_crop.rename(columns={'text': 'annotation'}, inplace=True)
    spotter_labels_crop = retain_alphabetic_annotations_only(spotter_labels_crop)

    IoU_matrix = calculate_IoU_matrix(list(spotter_labels_crop[['label_polygons', 'annotation']].itertuples(index=False, name=None)), list(gt_labels_crop[['label_polygons', 'annotation']].itertuples(index=False, name=None)))
    num_detected = IoU_matrix.shape[0]
    num_gt = IoU_matrix.shape[1]
    IoU_pairs = maximize_1to1_IoU(IoU_matrix)
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

def prec_rec(IoU_pairs, num_detected, num_gt):
    IoU_pairs = pd.DataFrame(IoU_pairs, columns=['geo_IoU', 'spotter_txt', 'gt_txt'])
    print("Avg of Geographic Precision: " + str(IoU_pairs['geo_IoU'].astype(float).sum(axis=0) / num_detected))
    print("Avg of Geographic Recall: " + str(IoU_pairs['geo_IoU'].astype(float).sum(axis=0) / num_gt))
    IoU_pairs['normalized_txt_similarity'] = IoU_pairs.apply(lambda row: text_compare(row['spotter_txt'], row['gt_txt']), axis=1)
    print("Avg of Text Precision: " + str(IoU_pairs['normalized_txt_similarity'].astype(float).sum(axis=0) / num_detected))
    print("Avg of Text Recall: " + str(IoU_pairs['normalized_txt_similarity'].astype(float).sum(axis=0) / num_gt))