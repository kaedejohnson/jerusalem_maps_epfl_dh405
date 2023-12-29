import random
import math
import numpy as np
from shapely.geometry import Polygon, MultiPolygon
from collections import Counter
import json 
from itertools import combinations

# Flatten stacked subsets    
def intersection_over_minimum(obj1, obj2):
    if (isinstance(obj1, Polygon) or isinstance(obj1, MultiPolygon)) and (isinstance(obj2, Polygon) or isinstance(obj2, MultiPolygon)):
        IoM = obj1.intersection(obj2).area / min(obj1.area, obj2.area)
    elif isinstance(obj1, str) and isinstance(obj2, str):
        obj1 = obj1.lower()
        obj2 = obj2.lower()
        cntr1 = Counter(obj1)
        cntr2 = Counter(obj2)
        global_char_set = set(cntr1.keys()) | set(cntr2.keys())
        IoM = sum(min(cntr1[char], cntr2[char]) for char in global_char_set) / min(len(obj1), len(obj2))
    else:
        print(obj1, obj2)
        print("both inputs must be of the same type (Polygon or string)")
        IoM = np.nan
    return IoM

def IoMs(label1, label2):
    poly1 = label1[0]
    text1 = label1[1]
    poly2 = label2[0]
    text2 = label2[1]
    if len(set(text1.lower()) | set(text2.lower())) == 0:
        return (0, 0)
    if not poly1.intersects(poly2):
        return (0, 0)
    poly_IoU = intersection_over_minimum(poly1, poly2)
    text_IoU = intersection_over_minimum(text1, text2)
    return (poly_IoU, text_IoU)

def update_P_matrix(df_w_labels, geo_threshold, text_threshold, P = None):
    labels = df_w_labels['labels']
    if P == None:
        P = {}
    to_combine = []
    for i, j in combinations(labels.index, 2):
        if i not in P.keys():
            P[i] = {}
        if j in P[i].keys():
            pass
        else:
            similarity = IoMs(labels[i], labels[j])
            P[i][j] = similarity
            if similarity[0] > geo_threshold and similarity[1] > text_threshold:
                to_combine.append((i,j))
    return P, to_combine

def combine_labels(label1, label2):
    poly1 = label1[0]
    text1 = label1[1]
    poly2 = label2[0]
    text2 = label2[1]
    poly_new = poly1.union(poly2)
    text_new = text1 if len(text1) > len(text2) else text2
    return (poly_new, text_new)

def amalgamate_labels(df, P, to_combine):
    for pair in to_combine:
        if str(pair[0]) in df.index and str(pair[1]) in df.index:
            new_label = combine_labels(df.loc[str(pair[0])]['labels'], df.loc[str(pair[1])]['labels'])
            df.loc[str(int(df.index[-1]) + 1)] = [new_label]
            df = df.drop([str(pair[0])]).copy()
            df = df.drop([str(pair[1])]).copy()
            try:
                P.pop(pair[0])
            except:
                pass
            try:
                P.pop(pair[1])
            except:
                pass
        else: # one of the polygons has already been swallowed so no combination can no longer occur
            pass
    return df, P

def amalgamate_labels_wrapper(df, geo_threshold, text_threshold):
    pre_amal = 0
    post_amal = len(df)
    P = None
    while pre_amal - post_amal != 0:
        pre_amal = post_amal
        P, to_combine = update_P_matrix(df, geo_threshold, text_threshold, P)
        print(str(pre_amal) + " labels.")
        df, P = amalgamate_labels(df, P, to_combine)
        post_amal = len(df)
    print("Amalgamation completed with " + str(pre_amal) + " labels.")
    return df