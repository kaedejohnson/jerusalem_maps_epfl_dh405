import json
import Clustering
import TextRectify
import shapely as sh
from shapely.geometry import Polygon, MultiPolygon


def subword_deduplication(map_name_in_strec, do_cluster_pre_merge = True):
    # Find duplicates
    with open(f'processed/strec/{map_name_in_strec}/combined_tagged_all_layers.json', 'r', encoding='utf-8') as f:

        clustered = Clustering.cluster_polygons(json.load(f))

        # visualize clusters
        #image = Clustering.visualize_polygons(clustered, f'processed/strec/{map_name_in_strec}/raw.jpeg')
        #image.save(f'processed/strec/{map_name_in_strec}/combined_tagged_all_layers_rectified.jpeg')
    
    # Text rectification
    for label, cluster in clustered.items():
        texts = []
        scores = []
        for polygon in cluster:
            texts.append(polygon['text'])
            scores.append(polygon['score'])

        rectifier = TextRectify.TextRectifier(0.95, 0.5, 10, True, True)

        rectifier.feed_data(texts, scores)

        rectifier.fit()

        rectified, mask = rectifier.get_rectified_text()

        if rectified is None:
            rectified = max(texts, key=len)

        for i in range(len(cluster)):
            cluster[i]['text'] = rectified[i]
            cluster[i]['keep'] = mask[i]


    polygon_x = {}
    polygon_y = {}
    texts = {}
    scores = {}
    i = 0
    for label, cluster in clustered.items():
        if do_cluster_pre_merge:
            polygons = []
            for polygon in cluster:
                poly = Polygon([(x,y) for x,y in zip(polygon['polygon_x'], polygon['polygon_y'])]).buffer(0)
                polygons.append(poly)

            p_merged = polygons[0]
            for p in polygons[1:]:
                p_merged = p_merged.union(p)
            
            p_merged_x = []
            p_merged_y = []
            if isinstance(p_merged, sh.geometry.polygon.Polygon):
                p_merged_x = p_merged.exterior.coords.xy[0]
                p_merged_y = p_merged.exterior.coords.xy[1]
            elif isinstance(p_merged, sh.geometry.multipolygon.MultiPolygon):
                for p in p_merged.geoms: # kaede added .geoms - package version differences
                    p_x = p.exterior.coords.xy[0]
                    p_y = p.exterior.coords.xy[1]
                    p_merged_x.extend(p_x)
                    p_merged_y.extend(p_y)
                    
            p_merged_x = list(p_merged_x)
            p_merged_y = list(p_merged_y)
            
            polygon_x[str(i)] = p_merged_x
            polygon_y[str(i)] = p_merged_y
            texts[str(i)] = cluster[0]['text']
            scores[str(i)] = cluster[0]['score']
            i += 1
        else:
            for polygon in cluster:
                polygon_x[str(i)] = polygon['polygon_x']
                polygon_y[str(i)] = polygon['polygon_y']
                texts[str(i)] = polygon['text']
                scores[str(i)] = polygon['score']
                i += 1

    print(f"{i} polygons kept.")
    json_data = {'polygon_x': polygon_x, 'polygon_y': polygon_y, 'text': texts, 'score': scores}

    with open(f'processed/strec/{map_name_in_strec}/combined_tagged_all_layers_rectified_premerge.json', 'w', encoding='utf-8') as f:
        json.dump(json_data, f, ensure_ascii=False, indent=4)