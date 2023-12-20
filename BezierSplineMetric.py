from itertools import combinations
import BezierSpline
import numpy as np
import SpotterWrapper
from PIL import Image, ImageFile

def calc_neighbours(df, radius_multiplier = 40, texts = None):
    pca_features = df['PCA_features']

    neighbours = [[] for i in range(len(pca_features))]
    multiplier = radius_multiplier
    for i, j in combinations(range(len(pca_features)), 2):
        c_i = pca_features[i][0]['Centroid']
        v_i = pca_features[i][0]['PCA_Var'][1] + pca_features[i][0]['PCA_Var'][0]
        c_j = pca_features[j][0]['Centroid']
        v_j = pca_features[j][0]['PCA_Var'][1] + pca_features[j][0]['PCA_Var'][0]
        
        dist_sqr = (c_i[0] - c_j[0])**2 + (c_i[1] - c_j[1])**2

        if dist_sqr < multiplier * v_i:
            neighbours[i].append(j)
        if dist_sqr < multiplier * v_j:
            neighbours[j].append(i)
    df['neighbours'] = neighbours
    return df

def spline_metric_for_anchor_pair(anchor_pair, i_switchable, j_switchable):
    splines = []
    spline_plain = BezierSpline.BezierSpline()
    spline_plain.from_pca(anchor_pair[0], anchor_pair[1], std_var_factor = 1)
    splines.append(spline_plain)
    if i_switchable == True:
        spline_switch_i = BezierSpline.BezierSpline()
        spline_switch_i.from_pca(anchor_pair[0], anchor_pair[1], 1, 1, 0)
        splines.append(spline_switch_i)
    if j_switchable == True:
        spline_switch_j = BezierSpline.BezierSpline()
        spline_switch_j.from_pca(anchor_pair[0], anchor_pair[1], 1, 0, 1)
        splines.append(spline_switch_j)
    if i_switchable == True and j_switchable == True:
        spline_switch_both = BezierSpline.BezierSpline()
        spline_switch_both.from_pca(anchor_pair[0], anchor_pair[1], 1, 1, 1)
        splines.append(spline_switch_both)

    anchor_dist = 1 # np.linalg.norm(best_pair[0]['Centroid'] - best_pair[1]['Centroid'])

    inner_min_cuvature = 10000000
    for spline in splines:
        max_curvature = spline.get_max_curvature(20)  # Size invariant curvature with distance penalty
        gap_penalty = spline.get_control_seg_length(0, 1)
        max_cost = max_curvature * anchor_dist * gap_penalty
        if max_cost < inner_min_cuvature:
            inner_min_cuvature = max_cost
            inner_best_spline = spline

    #if inner_min_cuvature < min_cuvature:
    #    min_cuvature = inner_min_cuvature
    #    best_spline = inner_best_spline
    
    return inner_best_spline, inner_min_cuvature

def spline_metric(df, texts = None):
    b_splines = []
    all_splines = []
    scores = []
    pca_features = df['PCA_features']
    neighbours = df['neighbours']

    for i in range(len(pca_features)):
        pca_i = pca_features[i]
        i_switchable = False
        i_has_sub_anchor = False
        if len(pca_i) == 3:
            pca_i = pca_i[1:]
            i_has_sub_anchor = True
        else:
            if pca_i[0]['PCA_Expands'][0] < 1.5 * pca_i[0]['PCA_Expands'][1]:
                i_switchable = True

        min_cuvature = 10000000
        best_spline = None

        curr_i_score_dict = {}
        curr_i_spline_list = []

        for j in neighbours[i]:
            pca_j = pca_features[j]
            j_switchable = False
            j_has_sub_anchor = False
            if len(pca_j) == 3:
                pca_j = pca_j[1:]
                j_has_sub_anchor = True
            else:
                if pca_j[0]['PCA_Expands'][0] < 1.5 * pca_j[0]['PCA_Expands'][1]:
                    j_switchable = True

            for a_i, anchor_i in enumerate(pca_i):
                a_id_i = (i, a_i)
                #if i_has_sub_anchor == True:
                #    a_id_i = (i, a_i + 1)
                curr_i_score_dict[a_id_i[1]] = {}
                for a_j, anchor_j in enumerate(pca_j):
                    a_id_j = (j, a_j)
                    #if j_has_sub_anchor == True:
                    #    a_id_j = (j, a_j + 1)
                    pair = (anchor_i, anchor_j)
                    spline, score = spline_metric_for_anchor_pair(pair, i_switchable, j_switchable)
                    curr_i_spline_list.append(spline, (a_id_i, a_id_j))
                    curr_i_score_dict[a_id_i[1]][a_id_j] = score
                

        all_splines.append(curr_i_spline_list)
        scores.append(curr_i_score_dict)

        if best_spline != None:
            b_splines.append([best_spline, min_cuvature])
        else:
            b_splines.append([None, -1])

    #df['b_splines'] = b_splines
    df['all_splines'] = all_splines
    df['bezier_scores'] = scores
    """
    scores = [
        {
            anchor_id1: {(poly_id1, anchor_id3): score, (poly_id3, anchor_id5): score, ...}, 
            anchor_id2: {(poly_id2, anchor_id4): score, (poly_id4, anchor_id6): score, ...},
            ...
        },
        ...
    ]
    """
    
    return df

def get_distance_metric(score_dict, i_anchors, j_anchors, infinitely_large_as):
    i_score = infinitely_large_as
    i_minimum_at = tuple()
    for anchor_id_i in i_anchors:
        for anchor_id_j in j_anchors:
            i_scores = score_dict[anchor_id_i[0]][anchor_id_i[1]]
            if anchor_id_j in i_scores.keys():
                score = i_scores[anchor_id_j]
                if score < i_score:
                    i_score = score
                    i_minimum_at = (anchor_id_i, anchor_id_j)

    j_score = infinitely_large_as
    j_minimum_at = tuple()
    for anchor_id_j in j_anchors:
        for anchor_id_i in i_anchors:
            j_scores = score_dict[anchor_id_j[0]][anchor_id_j[1]]
            if anchor_id_i in j_scores.keys():
                score = j_scores[anchor_id_i]
                if score < j_score:
                    j_score = score
                    j_minimum_at = (anchor_id_i, anchor_id_j)

    score = infinitely_large_as
    if i_score < j_score:
        return i_score, i_minimum_at
    else:
        return j_score, j_minimum_at

def draw_splines(map_name_in_strec, polygons, texts, PCA_features, all_splines, spline_metric_threshold):
    vis = SpotterWrapper.PolygonVisualizer()
    canvas = Image.open(f'processed/strec/{map_name_in_strec}/raw.jpeg')
    vis.canvas_from_image(canvas)
    vis.draw_poly(polygons, texts, PCA_features, [sp for sp in all_splines if sp[1] < spline_metric_threshold])
    vis.save(f'processed/strec/{map_name_in_strec}/visualized_splines.jpeg')