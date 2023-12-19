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

def spline_metric(df, texts = None):
    b_splines = []
    all_splines = []
    scores = []
    pca_features = df['PCA_features']
    neighbours = df['neighbours']

    for i in range(len(pca_features)):
        pca_i = pca_features[i]
        i_switchable = False
        if len(pca_i) == 3:
            pca_i = pca_i[1:]
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
            if len(pca_j) == 3:
                pca_j = pca_j[1:]
            else:
                if pca_j[0]['PCA_Expands'][0] < 1.5 * pca_j[0]['PCA_Expands'][1]:
                    j_switchable = True
            best_pair = None
            min_dist = 10000000
            for anchor_i in pca_i:
                for anchor_j in pca_j:
                    dist_sqr = (anchor_i['Centroid'][0] - anchor_j['Centroid'][0])**2 + (anchor_i['Centroid'][1] - anchor_j['Centroid'][1])**2
                    if dist_sqr < min_dist:
                        min_dist = dist_sqr
                        best_pair = (anchor_i, anchor_j)
            
            if best_pair != None:
                splines = []
                spline_plain = BezierSpline.BezierSpline()
                spline_plain.from_pca(best_pair[0], best_pair[1], std_var_factor = 1)
                splines.append(spline_plain)
                if i_switchable == True:
                    spline_switch_i = BezierSpline.BezierSpline()
                    spline_switch_i.from_pca(best_pair[0], best_pair[1], 1, 1, 0)
                    splines.append(spline_switch_i)
                if j_switchable == True:
                    spline_switch_j = BezierSpline.BezierSpline()
                    spline_switch_j.from_pca(best_pair[0], best_pair[1], 1, 0, 1)
                    splines.append(spline_switch_j)
                if i_switchable == True and j_switchable == True:
                    spline_switch_both = BezierSpline.BezierSpline()
                    spline_switch_both.from_pca(best_pair[0], best_pair[1], 1, 1, 1)
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
                    curr_i_spline_list.append([spline, max_cost])

                if inner_min_cuvature < min_cuvature:
                    min_cuvature = inner_min_cuvature
                    best_spline = inner_best_spline
                
                curr_i_score_dict[j] = min_cuvature

                #if (i == 1026 and j == 1030) or (i == 1030 and j == 1026):
                #    print(i, j)
                #    best_spline.draw()

        all_splines.append(curr_i_spline_list)
        scores.append(curr_i_score_dict)

        if best_spline != None:
            b_splines.append([best_spline, min_cuvature])
        else:
            b_splines.append([None, -1])

    #df['b_splines'] = b_splines
    #df['all_splines'] = all_splines
    df['scores'] = scores
    
    return df

def get_distance_metric(df, i, j, infinitely_large_as):
    i_has_j = False
    j_has_i = False

    i_scores = df.loc[i]['scores']
    j_scores = df.loc[j]['scores']
    if j in i_scores.keys():
        i_has_j = True
    elif i in j_scores.keys():
        j_has_i = True

    if i_has_j == True and j_has_i == True:
        return min(i_scores[j], j_scores[i])
    elif i_has_j == True:
        return i_scores[j]
    elif j_has_i == True:
        return j_scores[i]
    else:
        return infinitely_large_as # Infinitely large distance

def draw_splines(map_name_in_strec, polygons, texts, PCA_features, all_splines):
    vis = SpotterWrapper.PolygonVisualizer()
    canvas = Image.open(f'processed/strec/{map_name_in_strec}/raw.jpeg')
    vis.canvas_from_image(canvas)
    vis.draw_poly(polygons, texts, PCA_features, [sp for sp in all_splines if sp[1] < 0.5])
    vis.save(f'processed/strec/{map_name_in_strec}/visualized_splines.jpeg')