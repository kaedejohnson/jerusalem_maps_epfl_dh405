import numpy as np
from sklearn.decomposition import PCA
import shapely as sh
from itertools import combinations

def calc_PCA_feats(polygons, do_separation = True, enhance_coords = True):
    PCA_features = []
    for poly in polygons:
        _coords = []
        if isinstance(poly, sh.geometry.polygon.Polygon):
            polygon_x = poly.exterior.coords.xy[0]
            polygon_y = poly.exterior.coords.xy[1]
            for x, y in zip(polygon_x, polygon_y):
                _coords.append([x, y])
        elif isinstance(poly, sh.geometry.multipolygon.MultiPolygon):
            for p in poly:
                polygon_x = p.exterior.coords.xy[0]
                polygon_y = p.exterior.coords.xy[1]
                for x, y in zip(polygon_x, polygon_y):
                    _coords.append([x, y])
        
        coords = _coords
        if enhance_coords == True:
            for a, b in combinations(_coords, 2):
                coords.append([(a[0] + b[0])/2.0, (a[1] + b[1])/2.0])

        _pca = PCA(n_components = 2)
        _pca.fit(coords)
        V = _pca.explained_variance_
        C = _pca.components_

        if (do_separation == False):
            centroid = poly.centroid
            PCA_features.append([{'Centroid': centroid,'PCA_Var': V, 'PCA_Basis': C}])
        else:
            if V[0] < 4 * V[1]:
                centroid = poly.centroid
                PCA_features.append([{'Centroid': centroid,'PCA_Var': V, 'PCA_Basis': C}])
            else:
                U = _pca.transform(coords)[:, 0]
                A_coords = []
                B_coords = []
                for coord, u in zip(coords, U):
                    if u > 0:
                        A_coords.append(coord)
                    else:
                        B_coords.append(coord)

                _pca_A = PCA(n_components = 2)
                _pca_A.fit(A_coords)
                V_A = _pca_A.explained_variance_
                C_A = _pca_A.components_
                centroid_A = np.mean(A_coords, axis=0)

                _pca_B = PCA(n_components = 2)
                _pca_B.fit(B_coords)
                V_B = _pca_B.explained_variance_
                C_B = _pca_B.components_
                centroid_B = np.mean(B_coords, axis=0)
                PCA_features.append([{'Centroid': centroid_B,'PCA_Var': V_B, 'PCA_Basis': C_B}, {'Centroid': centroid_A,'PCA_Var': V_A, 'PCA_Basis': C_A}])
    
    return PCA_features