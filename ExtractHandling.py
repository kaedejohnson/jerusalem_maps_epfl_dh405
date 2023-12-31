import pandas as pd
import pickle
from shapely.geometry import Polygon, MultiPolygon
import json 

# Load in labels extracted by spotter and prepare them for input into different stages
def load_spotter_labels(map_name_in_strec, layer_json_w_extension):
    with open('processed/strec/' + map_name_in_strec + '/' + layer_json_w_extension) as f:
        spotter_labels_tmp = json.load(f)
    spotter_labels = pd.DataFrame(spotter_labels_tmp)
    spotter_labels = spotter_labels.rename(columns={'polygon_x':'all_points_x','polygon_y':'all_points_y'})
    return spotter_labels

## Convert labels to polygon objects (df needs all_points_x and all_points_y cols)
def create_polygon_object(x_coords, y_coords):
    return Polygon(zip(x_coords, y_coords)).buffer(0) # buffer fixes errors that occur due to weird self-overlapping edges when manually labeling

def cast_coords_as_Polygons(df):
    df_copy = df.copy()
    df_copy.loc[:, 'label_polygons'] = df.apply(lambda row: create_polygon_object(row['all_points_x'], row['all_points_y']), axis=1)
    return df_copy   

def load_processed_labels(map_name_in_strec, methods):
    if methods == "methods_1_2":
        df = pickle.load(open('processed/strec/' + map_name_in_strec + '/deduplicated_flattened_labels.pickle', 'rb'))
    elif methods == "methods_1_2_3":
        df = pickle.load(open('processed/strec/' + map_name_in_strec + '/fully_processed_labels.pickle', 'rb'))
    elif methods == "methods_1_2_r":
        df = pickle.load(open('processed/strec/' + map_name_in_strec + '/fully_processed_labels_rumsey.pickle', 'rb'))
    elif methods == "methods_1_2_3a":
        df = pickle.load(open('processed/strec/' + map_name_in_strec + '/fully_processed_labels_altslsr.pickle', 'rb'))
    df = pd.DataFrame(df['labels'].tolist(), columns=['label_polygons','annotation'])
    return df

def prepare_labels_for_amalgamation(map_name_in_strec):
    df = load_spotter_labels(map_name_in_strec, "combined_tagged_all_layers_rectified_premerge.json")
    df = cast_coords_as_Polygons(df)
    df['labels'] = df.apply(lambda row: (row['label_polygons'], row['text']), axis=1)
    df = df[['labels']]
    return df
