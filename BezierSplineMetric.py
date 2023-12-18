from itertools import combinations
import BezierSpline
import numpy as np

def calc_neighbours(polygons, PCA_features, radius_multiplier = 60):
    neighbours = [[] for i in range(len(polygons))]
    multiplier = radius_multiplier
    for i, j in combinations(range(len(polygons)), 2):
        c_i = PCA_features[i][0]['Centroid']
        v_i = PCA_features[i][0]['PCA_Var'][1] + PCA_features[i][0]['PCA_Var'][0]
        c_j = PCA_features[j][0]['Centroid']
        v_j = PCA_features[j][0]['PCA_Var'][1] + PCA_features[j][0]['PCA_Var'][0]
        
        dist_sqr = (c_i[0] - c_j[0])**2 + (c_i[1] - c_j[1])**2

        if dist_sqr < multiplier * v_i:
            neighbours[i].append(j)
        if dist_sqr < multiplier * v_j:
            neighbours[j].append(i)

    return neighbours

def spline_metric(polygons, PCA_features, neighbours):
    b_splines = []
    all_splines = []
    scores = [{} for _ in range(len(polygons))]
    for i in range(len(polygons)):
        pca_i = PCA_features[i]
        i_switchable = False
        if len(pca_i) == 3:
            pca_i = pca_i[1:]
        else:
            if pca_i[0]['PCA_Expands'][0] < 1.5 * pca_i[0]['PCA_Expands'][1]:
                i_switchable = True

        min_cuvature = 10000000
        best_spline = None
        for j in neighbours[i]:
            pca_j = PCA_features[j]
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

                anchor_dist = np.linalg.norm(best_pair[0]['Centroid'] - best_pair[1]['Centroid'])

                inner_min_cuvature = 10000000
                for spline in splines:
                    max_curvature = spline.get_max_curvature(20) * anchor_dist * spline.get_control_seg_length(0, 1) # Size invariant curvature with distance penalty
                    if max_curvature < inner_min_cuvature:
                        inner_min_cuvature = max_curvature
                        inner_best_spline = spline
                    all_splines.append([spline, max_curvature])

                if inner_min_cuvature < min_cuvature:
                    min_cuvature = inner_min_cuvature
                    best_spline = inner_best_spline
                
                scores[i][j] = min_cuvature
        
        if best_spline != None:
            b_splines.append([best_spline, min_cuvature])

    return b_splines, all_splines, scores

def get_distance_metric(score, i, j):
    if score[i][j] != None:
        return score[i][j]
    elif score[j][i] != None:
        return score[j][i]
    else:
        return 10000000
