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
        U = _pca.transform(coords)
        UX = U[:, 0]
        UY = U[:, 1]
        expand_major = max(UX.max(), -UX.min())
        expand_minor = max(UY.max(), -UY.min())

        centroid = np.array([poly.centroid.x, poly.centroid.y])
        pca_feature:list = [{'Centroid': centroid, 'PCA_Var': V, 'PCA_Basis': C, 'PCA_Expands': np.array([expand_major, expand_minor])}]
        if (do_separation == True):
            if V[0] < 4 * V[1]:
                pass
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
                U_A = _pca_A.transform(A_coords)
                UX_A = U_A[:, 0]
                UY_A = U_A[:, 1]
                expand_major_A = max(UX_A.max(), -UX_A.min())
                expand_minor_A = max(UY_A.max(), -UY_A.min())

                _pca_B = PCA(n_components = 2)
                _pca_B.fit(B_coords)
                V_B = _pca_B.explained_variance_
                C_B = _pca_B.components_
                centroid_B = np.mean(B_coords, axis=0)
                U_B = _pca_B.transform(B_coords)
                UX_B = U_B[:, 0]
                UY_B = U_B[:, 1]
                expand_major_B = max(UX_B.max(), -UX_B.min())
                expand_minor_B = max(UY_B.max(), -UY_B.min())

                pca_feature.extend([{'Centroid': centroid_B,'PCA_Var': V_B, 'PCA_Basis': C_B, 'PCA_Expands': np.array([expand_major_B, expand_minor_B])}, {'Centroid': centroid_A,'PCA_Var': V_A, 'PCA_Basis': C_A,  'PCA_Expands': np.array([expand_major_A, expand_minor_A])}])
        
        PCA_features.append(pca_feature)

    return PCA_features

def crop_image_with_nabb(original_image, centroid, angle, expand_major, expand_minor):
    expand_major_rotated = np.array([expand_major * np.cos(np.deg2rad(angle)), expand_major * np.sin(np.deg2rad(angle))])
    expand_minor_rotated = np.array([expand_minor * np.sin(np.deg2rad(angle)), -expand_minor * np.cos(np.deg2rad(angle))])
    nabb_coordinates = [centroid + expand_major_rotated + expand_minor_rotated,
                        centroid - expand_major_rotated + expand_minor_rotated,
                        centroid - expand_major_rotated - expand_minor_rotated,
                        centroid + expand_major_rotated - expand_minor_rotated]
    nabb_coordinates = [x[0] for x in nabb_coordinates]
    # Create a shapely polygon from the NABB coordinates
    polygon = Polygon(nabb_coordinates)

    # Get the bounding box of the polygon
    bounding_box = polygon.bounds

    # Convert bounding box coordinates to integers
    bounding_box = tuple(map(int, bounding_box))

    # Crop the image using the bounding box
    cropped_image = original_image.crop(bounding_box)

    # Rotate the cropped image
    cropped_image = cropped_image.rotate(angle, expand=True)
    
    cropped_image_center = np.array(cropped_image.size) / 2
    crop_region = [cropped_image_center[0] - expand_major, cropped_image_center[1] - expand_minor,
                     cropped_image_center[0] + expand_major, cropped_image_center[1] + expand_minor]

    # Crop the image again using the crop region
    cropped_image = cropped_image.crop(crop_region)

    return cropped_image

def polygon_crop(poly, image, enhance_coords = True):
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
    U = _pca.transform(coords)
    UX = U[:, 0]
    UY = U[:, 1]
    expand_major = max(UX.max(), -UX.min())
    expand_minor = max(UY.max(), -UY.min())
    centroid = np.array(poly.centroid.coords)
    angle = np.rad2deg(np.arctan2(C[0, 1], C[0, 0]))
    return crop_image_with_nabb(image, centroid, angle, expand_major, expand_minor)
