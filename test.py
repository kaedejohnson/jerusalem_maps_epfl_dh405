from PIL import Image, ImageDraw
from shapely.geometry import Polygon
import matplotlib.pyplot as plt
import numpy as np
import shapely as sh
from shapely.geometry import Polygon, Point
from shapely.ops import triangulate
from itertools import combinations
from sklearn.decomposition import PCA


def crop_image_with_nabb(original_image, centroid, angle, expand_major, expand_minor):
    expand_major_rotated = np.array([expand_major * np.cos(np.deg2rad(angle)), expand_major * np.sin(np.deg2rad(angle))])
    expand_minor_rotated = np.array([expand_minor * np.sin(np.deg2rad(angle)), -expand_minor * np.cos(np.deg2rad(angle))])

    nabb_coordinates = [centroid + expand_major_rotated + expand_minor_rotated,
                        centroid - expand_major_rotated + expand_minor_rotated,
                        centroid - expand_major_rotated - expand_minor_rotated,
                        centroid + expand_major_rotated - expand_minor_rotated]
    
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
    plt.imshow(cropped_image)
    plt.show()

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
    centroid = np.array(poly.centroid)
    angle = np.rad2deg(np.arctan2(C[0, 1], C[0, 0]))
    return crop_image_with_nabb(image, centroid, angle, expand_major, expand_minor)


# Example usage:
image_path = "test.jpg"
output_path = "cropped_image.jpg"
original_image = Image.open(image_path)
poly = Polygon([(0, 100), (50, 150), (100, 200), (150, 250), (200, 300), (250, 350), (300, 400), (350, 350), (400, 300), (350, 250), (300, 200), (250, 150), (200, 100), (150, 50), (100, 0), (50, 50) ])
cropped_image = polygon_crop(poly, original_image)

# Plot the cropped image
plt.imshow(cropped_image)
plt.show()
