from shapely.geometry import Polygon

polygon1 = Polygon([(0, 0), (0, 2), (2, 2), (2, 0)])
polygon2 = Polygon([(1, 1), (1, 3), (3, 3), (3, 1)])
print(polygon1.intersection(polygon2))