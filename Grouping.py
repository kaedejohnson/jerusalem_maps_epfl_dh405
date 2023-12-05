import numpy as np
from sklearn.decomposition import PCA
import shapely as sh
from shapely.geometry import Polygon, Point
from shapely.ops import triangulate
from itertools import combinations
import random

def generate_samples(polygon, num_samples):
    triangles = triangulate(polygon)
    triangle_areas = [triangle.area for triangle in triangles]
    total_area = sum(triangle_areas)
    probabilities = [area / total_area for area in triangle_areas]
    
    samples = []
    for _ in range(num_samples):
        selected_triangle = random.choices(triangles, probabilities)[0]
        alpha = random.uniform(0, 1)
        beta = random.uniform(0, 1 - alpha)
        x_coords = selected_triangle.exterior.coords.xy[0][:3]
        y_coords = selected_triangle.exterior.coords.xy[1][:3]
        coords = np.array([[x, y] for x, y in zip(x_coords, y_coords)])
        point = (
            (1 - alpha - beta) * coords[0] +
            alpha * coords[1] +
            beta * coords[2]
        )
        samples.append(point)

    return samples

def calc_PCA_feats(polygons, do_separation = True, enhance_coords = True):
    PCA_features = []
    for poly in polygons:
        raw_coords = []
        if isinstance(poly, sh.geometry.polygon.Polygon):
            polygon_x = poly.exterior.coords.xy[0]
            polygon_y = poly.exterior.coords.xy[1]
            for x, y in zip(polygon_x, polygon_y):
                raw_coords.append(np.array([x, y]))
        elif isinstance(poly, sh.geometry.multipolygon.MultiPolygon):
            for p in poly:
                polygon_x = p.exterior.coords.xy[0]
                polygon_y = p.exterior.coords.xy[1]
                for x, y in zip(polygon_x, polygon_y):
                    raw_coords.append(np.array([x, y]))
        
        coords = raw_coords
        if enhance_coords == True: # Try to sample in the whole polygon
            #samples = generate_samples(Polygon(coords), 200)
            #coords = samples
            for a, b in combinations(coords, 2):
                coords.append((a + b) / 2)
                coords.append((a + 2*b) / 3)
                coords.append((2*a + b) / 3)

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
                raw_A_coords = []
                raw_B_coords = []
                for coord, u in zip(coords, U):
                    if u > 0:
                        raw_A_coords.append(coord)
                    else:
                        raw_B_coords.append(coord)

                A_coords = raw_A_coords
                B_coords = raw_B_coords

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