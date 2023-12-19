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
            for p in poly.geoms: # kaede added .geoms - package version differences
                polygon_x = p.exterior.coords.xy[0]
                polygon_y = p.exterior.coords.xy[1]
                for x, y in zip(polygon_x, polygon_y):
                    raw_coords.append(np.array([x, y]))
        
        tail_length = [np.linalg.norm(raw_coords[i+1] - raw_coords[i]) for i in range(len(raw_coords)-1)]
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
        expand_major = 0.5 * (UX.max()-UX.min())
        expand_minor = 0.5 * (UY.max()-UY.min())

        centroid = np.array([0.0,0.0])
        L = 0
        for coord, l in zip(coords, tail_length):
            centroid += coord * l
            L += l
        centroid /= L
        pca_feature:list = [{'Centroid': centroid, 'PCA_Var': V, 'PCA_Basis': C, 'PCA_Expands': np.array([expand_major, expand_minor])}]
        if (do_separation == True):
            if V[0] < 6 * V[1]:
                pass
            else:
                U = _pca.transform(coords)[:, 0]
                raw_A_coords = []
                tail_length_A = []
                raw_B_coords = []
                tail_length_B = []
                for coord, tl, u in zip(coords, tail_length, U):
                    if u > 0:
                        raw_A_coords.append(coord)
                        tail_length_A.append(tl)
                    else:
                        raw_B_coords.append(coord)
                        tail_length_B.append(tl)

                A_coords = raw_A_coords
                B_coords = raw_B_coords

                _pca_A = PCA(n_components = 2)
                _pca_A.fit(A_coords)
                V_A = _pca_A.explained_variance_
                C_A = _pca_A.components_
                centroid_A = np.array([0.0,0.0])
                L = 0
                for a_coord, l in zip(A_coords, tail_length_A):
                    centroid_A += a_coord * l
                    L += l
                centroid_A /= L
                U_A = _pca_A.transform(A_coords)
                UX_A = U_A[:, 0]
                UY_A = U_A[:, 1]
                expand_major_A = 0.5 * (UX_A.max()-UX_A.min())
                expand_minor_A = 0.5 * (UY_A.max()-UY_A.min())

                _pca_B = PCA(n_components = 2)
                _pca_B.fit(B_coords)
                V_B = _pca_B.explained_variance_
                C_B = _pca_B.components_
                centroid_B = np.array([0.0,0.0])
                L = 0
                for b_coord, l in zip(B_coords, tail_length_B):
                    centroid_B += b_coord * l
                    L += l
                centroid_B /= L
                U_B = _pca_B.transform(B_coords)
                UX_B = U_B[:, 0]
                UY_B = U_B[:, 1]
                expand_major_B = 0.5 * (UX_B.max()-UX_B.min())
                expand_minor_B = 0.5 * (UY_B.max()-UY_B.min())

                pca_feature.extend([{'Centroid': centroid_B,'PCA_Var': V_B, 'PCA_Basis': C_B, 'PCA_Expands': np.array([expand_major_B, expand_minor_B])}, {'Centroid': centroid_A,'PCA_Var': V_A, 'PCA_Basis': C_A,  'PCA_Expands': np.array([expand_major_A, expand_minor_A])}])
        
        PCA_features.append(pca_feature)
    refine_PCA_basis(PCA_features, polygons)
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
        for p in poly.geoms: # kaede added .geoms - package version differences
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

def residual_vec_x(params, dirs):
    R = []
    for d in dirs:
        a_i = d[0]
        b_i = d[1]
        x = params[0]
        c = abs(a_i * x + b_i * np.sqrt(1 - x ** 2))
        r = c * (1 - c)
        R.append(r)
    return np.array(R)

def jacobian_mat_x(params, dirs):
    J = []
    for d in dirs:
        x = params[0]
        y = np.sqrt(1 - x ** 2)
        #y = params[1]
        a_i = d[0]
        b_i = d[1]
        c = a_i * x + b_i * y
        #derivative_x = a_i - 2 * a_i ** 2 * x - a_i * b_i * y
        #derivative_y = b_i - a_i * b_i * x - 2 * b_i ** 2 * y
        #J.append([derivative_x, derivative_y])
        derivative = a_i - b_i * x / y
        if c > 0:
            J.append([derivative])
        else:
            J.append([-derivative])
    return np.array(J)

def residual_vec_y(params, dirs):
    R = []
    for d in dirs:
        a_i = d[0]
        b_i = d[1]
        y = params[0]
        c = a_i * np.sqrt(1 - y ** 2) + b_i * y
        r = c * (1 - c)
        R.append(r)
    return np.array(R)

def jacobian_mat_y(params, dirs):
    J = []
    for d in dirs:
        y = params[0]
        x = np.sqrt(1 - y ** 2)
        #y = params[1]
        a_i = d[0]
        b_i = d[1]
        c = a_i * x + b_i * y
        #derivative_x = a_i - 2 * a_i ** 2 * x - a_i * b_i * y
        #derivative_y = b_i - a_i * b_i * x - 2 * b_i ** 2 * y
        #J.append([derivative_x, derivative_y])
        derivative = b_i - a_i * y / x
        if c > 0:
            J.append([derivative])
        else:
            J.append([-derivative])
    return np.array(J)

def residual_vec_theta(params, dirs):
    R = []
    for d in dirs:
        a_i = d[0]
        b_i = d[1]
        theta = params[0]
        c = abs(a_i * np.cos(theta) + b_i * np.sin(theta))
        r = c * (1 - c)
        R.append(r)
    return np.array(R)

def jacobian_mat_theta(params, dirs):
    J = []
    for d in dirs:
        theta = params[0]
        a_i = d[0]
        b_i = d[1]
        c = a_i * np.cos(theta) + b_i * np.sin(theta)
        derivative = (1 - 2 * abs(c)) * (- a_i * np.sin(theta) + b_i * np.cos(theta))
        if c > 0:
            J.append([derivative])
        else:
            J.append([-derivative])
    return np.array(J)

def solve_GM(initial_params, directions, weights_vec = None):
    result_params = initial_params
    if weights_vec == None:
        weights_vec = np.ones(len(directions))
    for i in range(20):
        J = np.array([w * j for w, j in zip(weights_vec,jacobian_mat_theta(result_params, directions))])
        R = np.array([w * r for w, r in zip(weights_vec,residual_vec_theta(result_params, directions))])
        delta = np.linalg.inv(J.T @ J) @ J.T @ R
        result_params = result_params - delta
        if np.linalg.norm(delta) < 0.00001:
            break
    theta = result_params[0]
    result = np.array([np.cos(theta), np.sin(theta)])
    return result

def align_with_original(init_p_x, result_x):
    init_first_x = init_p_x
    norm_init_first_x = np.linalg.norm(init_p_x)
    first_x = result_x
    norm_first_x = np.linalg.norm(first_x)
    second_x = np.array([result_x[1], -result_x[0]])
    cos_init_first = abs(init_first_x.dot(first_x) / (norm_init_first_x * norm_first_x))

    if cos_init_first > 0.5 * np.sqrt(2):
        return [first_x, second_x]
    else:
        return [second_x, first_x]
    
def refine_PCA_basis(PCA_features, polygons):
    for poly_id in range(len(polygons)):
        poly = polygons[poly_id]
        raw_coords = []
        if isinstance(poly, sh.geometry.polygon.Polygon):
            polygon_x = poly.exterior.coords.xy[0]
            polygon_y = poly.exterior.coords.xy[1]
            for x, y in zip(polygon_x, polygon_y):
                raw_coords.append(np.array([x, y]))
        elif isinstance(poly, sh.geometry.multipolygon.MultiPolygon):
            for p in poly.geoms: # kaede added: package discrepancy
                polygon_x = p.exterior.coords.xy[0]
                polygon_y = p.exterior.coords.xy[1]
                for x, y in zip(polygon_x, polygon_y):
                    raw_coords.append(np.array([x, y]))
        midpoints = [ (raw_coords[i+1] + raw_coords[i]) / 2 for i in range(len(raw_coords)-1) ]
        directions = [ raw_coords[i+1] - raw_coords[i] for i in range(len(raw_coords)-1) ]
        weights = [ np.sqrt(np.linalg.norm(d)) for d in directions ]
        pricipal_components = PCA_features[poly_id][0]['PCA_Basis'][0]
        directions = [ d / np.linalg.norm(d) if d.dot(pricipal_components) >= 0 else -d / np.linalg.norm(d) for d in directions]
        
        if len(PCA_features[poly_id]) == 1:
            initial_params = np.array([ pricipal_components[0], pricipal_components[1]])
            init_theta = np.arctan2(initial_params[1], initial_params[0])
            
            result = solve_GM([init_theta], directions, weights)

            PCA_features[poly_id][0]['PCA_Basis'] = align_with_original(initial_params, result)

        if len(PCA_features[poly_id]) == 3:
            pca_a = PCA_features[poly_id][1]
            pc_a = pca_a['PCA_Basis'][0]
            init_p_a = np.array([pc_a[0], pc_a[1]])
            init_theta_a = np.arctan2(init_p_a[1], init_p_a[0])
            pca_b = PCA_features[poly_id][2]
            pc_b = pca_b['PCA_Basis'][0]
            init_p_b = np.array([pc_b[0], pc_b[1]])
            init_theta_b = np.arctan2(init_p_b[1], init_p_b[0])

            center_a = pca_a['Centroid']
            
            # Get 8 closets midpoints to the center of the first PCA
            a_midpoints_ids = sorted(range(len(midpoints)), key = lambda x: np.linalg.norm(midpoints[x] - center_a))[:8]
            weights_a = [w for w, m in zip(weights, range(len(midpoints))) if m in a_midpoints_ids ]
            dirs_a = [d for d, m in zip(directions, range(len(midpoints))) if m in a_midpoints_ids ]
            weights_b = [w for w, m in zip(weights, range(len(midpoints))) if m not in a_midpoints_ids]
            dirs_b = [d for d, m in zip(directions, range(len(midpoints))) if m not in a_midpoints_ids ]

            result_a = solve_GM([init_theta_a], dirs_a, weights_a)
            result_b = solve_GM([init_theta_b], dirs_b, weights_b)
            
            PCA_features[poly_id][1]['PCA_Basis'] = align_with_original(initial_params, result_a)
            PCA_features[poly_id][2]['PCA_Basis'] = align_with_original(initial_params, result_b)