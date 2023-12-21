from shapely.geometry import Polygon, MultiPolygon
from itertools import combinations
import importlib
import Clustering
import TextRectify
import TextAmalgamate
import ExtractHandling
import SpotterWrapper
import Grouping
import BezierSplineMetric
import FontSimilarity
import numpy as np

importlib.reload(SpotterWrapper)
importlib.reload(Grouping)
importlib.reload(Clustering)
importlib.reload(TextRectify)
importlib.reload(TextAmalgamate)
importlib.reload(ExtractHandling)
importlib.reload(BezierSplineMetric)
importlib.reload(FontSimilarity)


def combine_labels(label1_row, label2_row):

    poly1 = label1_row['labels'][0]
    poly2 = label2_row['labels'][0]
    poly_new = poly1.union(poly2) # returns multipolygon object with disjoint polygons if polygons are disjoint

    #text1 = label1_row['labels'][1]
    #text2 = label2_row['labels'][1]
    #leftmost_poly = [poly1, poly2].index(min([poly1, poly2], key=lambda shape: shape.bounds[0]))
    #if leftmost_poly == 0:
    #    text_new = text1 + " " + text2
    #else:
    #    text_new = text2 + " " + text1
    text_new = ''

    bezier_scores1 = label1_row['bezier_scores']
    bezier_scores2 = label2_row['bezier_scores']
    bezier_scores_new = {key: min(bezier_scores1.get(key, float('inf')), bezier_scores2.get(key, float('inf'))) for key in set(bezier_scores1) | set(bezier_scores2)}

    font_scores1 = label1_row['font_scores']
    font_scores2 = label2_row['font_scores']
    font_scores_new = {key: max(font_scores1.get(key, 0), font_scores2.get(key, 0)) for key in set(font_scores1) | set(font_scores2)}
    neighbours1 = label1_row['neighbours']
    neighbours2 = label2_row['neighbours']
    if neighbours1 is None:
        neighbours1 = []
    if neighbours2 is None:
        neighbours2 = []
    neighbours_new = list(set(neighbours1 + neighbours2))

    text_list1 = label1_row['text_list']
    text_list2 = label2_row['text_list']
    text_list_new = text_list1 + text_list2

    constituents1 = label1_row['constituents']
    constituents2 = label2_row['constituents']
    constituents_new = constituents1 + constituents2

    return [(poly_new, text_new), poly_new, text_new, neighbours_new, bezier_scores_new, font_scores_new, text_list_new, constituents_new]

def recover_sequence(df, R, to_combine):

    sorted_to_combine = sorted(to_combine, key=lambda x: x[1])

    for pair_w_score in sorted_to_combine:
        pair = pair_w_score[0]
        if pair[0] in df.index and pair[1] in df.index:
            new_label = combine_labels(df.loc[pair[0]], df.loc[pair[1]])
            new_label_index = int(df.index[-1]) + 1
            df.loc[new_label_index] = new_label
            df = df.drop([pair[0]]).copy()
            df = df.drop([pair[1]]).copy()
            try:
                R.pop(pair[0])
            except:
                pass
            try:
                R.pop(pair[1])
            except:
                pass
        else: # one of the polygons has already been recovered into a sequence so recovery can no longer occur
            pass
    return df, R

def update_R_matrix(df, font_threshold, bezier_threshold, R = None):
    if R == None:
        R = {}
    to_combine = []
    for i, j in combinations(df.index, 2):
        if i not in R.keys():
            R[i] = {}
        if j in R[i].keys():
            pass
        else:
            font_similarity_score = 1 # FontSimilarity.get_similarity_metric(df, i, j)
            spline_distance_score = BezierSplineMetric.get_distance_metric(df, i, j, infinitely_large_as=10000000)
            R[i][j] = (font_similarity_score, spline_distance_score)
            if font_similarity_score > font_threshold and spline_distance_score < bezier_threshold:
                to_combine.append(((i,j), spline_distance_score))
    return R, to_combine

def sl_sequence_recovery_wrapper(df, font_threshold, bezier_threshold):

    pre_seqrec = 0
    post_seqrec = len(df)
    R = None
    df['text_list'] = df.apply(lambda row: [(np.array([row['polygons'].centroid.x, row['polygons'].centroid.y]), row['texts'])], axis=1)
    df['constituents'] = df.apply(lambda row: [df.index.get_loc(row.name)], axis=1)
    while pre_seqrec - post_seqrec != 0:
        pre_seqrec = post_seqrec

        # map it to comparison matrix, find candidates for sequences
        R, to_combine = update_R_matrix(df, font_threshold, bezier_threshold, R)
        print(str(pre_seqrec) + " labels.")

        # recover sequences based on candidates
        df, R = recover_sequence(df, R, to_combine)
        post_seqrec = len(df)

    for index, row in df.iterrows():
        sorted_text = sorted(row['text_list'], key=lambda x: x[0][0])
        row['texts'] = " ".join([_text[1] for _text in sorted_text])
        row['labels'] = (row['labels'][0], row['texts'])


    print("Sequence Recovery completed with " + str(pre_seqrec) + " labels.")
    return df