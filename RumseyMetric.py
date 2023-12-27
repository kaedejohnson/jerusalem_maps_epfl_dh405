import numpy as np
import shapely as sh
from itertools import combinations

def calc_rumsey_metric(df):
    df_anchors = []
    df_pts = []
    df_width = []  
    for index, row in df.iterrows():
        poly = row['labels'][0]
        polygon = []
        if isinstance(poly, sh.geometry.polygon.Polygon):
            polygon_x = poly.exterior.coords.xy[0]
            polygon_y = poly.exterior.coords.xy[1]
            for x, y in zip(polygon_x, polygon_y):
                polygon.append(np.array([x, y]))
        elif isinstance(poly, sh.geometry.multipolygon.MultiPolygon):
            for p in poly.geoms: # kaede added .geoms - package version differences
                polygon_x = p.exterior.coords.xy[0]
                polygon_y = p.exterior.coords.xy[1]
                for x, y in zip(polygon_x, polygon_y):
                    polygon.append(np.array([x, y]))
        max_dist = 0
        anchors = []
        for p1, p2 in combinations(polygon, 2):
            dist = (p1-p2).dot(p1-p2)
            if dist > max_dist:
                max_dist = dist
                anchors = [p1, p2]

        df_anchors.append(anchors)
        df_pts.append(polygon)
        if len(row['PCA_features']) == 1:
            df_width.append(row['PCA_features'][0]['PCA_Expands'][1])
        else:
            df_width.append(row['PCA_features'][1]['PCA_Expands'][1])

    df['anchors'] = df_anchors
    df['pts'] = df_pts
    df['width'] = df_width
    return df


def rumsey_metric(df, i, j):
    anchors_i = df.loc[i]['anchors']
    anchors_j = df.loc[j]['anchors']

    pts_i = df.loc[i]['pts']
    pts_j = df.loc[j]['pts']

    width_i = df.loc[i]['width']
    width_j = df.loc[j]['width']
    width = 0.5 * (width_i + width_j)
    width_2 = width * width

    #anchor_mat_i = np.array(anchors_i)
    #pts_mat_i = np.array(pts_i).transpose()
    #r1 = np.any(np.dot(anchor_mat_i, pts_mat_i) < width_2)
    #if r1:
    #    return 1

    #anchor_mat_j = np.array(anchors_j)
    #pts_mat_j = np.array(pts_j).transpose()
    #r2 = np.any(np.dot(anchor_mat_j, pts_mat_j) < width_2)
    #if r2:
    #    return 1


    for anchor_j in anchors_j:
        for pt_i in pts_i:
            dist = (anchor_j-pt_i).dot(anchor_j-pt_i)
            if dist < width_2:
                return 1
            
    for anchor_i in anchors_i:
        for pt_j in pts_j:
            dist = (anchor_i-pt_j).dot(anchor_i-pt_j)
            if dist < width_2:
                return 1
    
    return 0