import Evaluation
import ExtractHandling
import Grouping
import importlib
importlib.reload(Grouping)
%load_ext autoreload
from PIL import Image

for map_name_in_strec in ["saunders_1874", "vandevelde_1846"]: #"kiepert_1845"
    gt_labels_full = Evaluation.load_ground_truth_labels(map_name_in_strec, "components")
    gt_labels_full = ExtractHandling.cast_coords_as_Polygons(gt_labels_full)

    # the original index in the dataframe should be the indentifying number
    for index, row in gt_labels_full.iterrows():
        img = Grouping.polygon_crop(row['label_polygons'], Image.open("processed/strec/" + map_name_in_strec + "/raw.jpeg"))
        img.save("dependencies/ground_truth_labels/cropped/" + map_name_in_strec + "_" + str(index) + ".jpeg")


import json
with open("dependencies/ground_truth_labels/ground_truth_labels_backup_preidchg.json", "r", encoding="utf-8") as f:
    data_list = json.load(f)["data"]
data_list[0]

# kiepert (old -> new)
kiepert_crop_dict = {}
kiepert_crop_dict[0] = 1
kiepert_crop_dict[1] = 2
kiepert_crop_dict[2] = 4
kiepert_crop_dict[4] = 6
kiepert_crop_dict[7] = 10
kiepert_crop_dict[9] = 13
kiepert_crop_dict[10] = 14
kiepert_crop_dict[12] = 17
kiepert_crop_dict[13] = 18
kiepert_crop_dict[19] = 23
kiepert_crop_dict[21] = 25
kiepert_crop_dict[29] = 35
kiepert_crop_dict[30] = 37
kiepert_crop_dict[31] = 38
kiepert_crop_dict[32] = 39
kiepert_crop_dict[35] = 43
kiepert_crop_dict[37] = 45
kiepert_crop_dict[40] = 49
kiepert_crop_dict[44] = 53
kiepert_crop_dict[49] = 59
kiepert_crop_dict[53] = 64
kiepert_crop_dict[56] = 68
kiepert_crop_dict[59] = 71
kiepert_crop_dict[61] = 0

# saunders (old -> new)
saunders_crop_dict = {}
saunders_crop_dict[1] = 4
saunders_crop_dict[3] = 13
saunders_crop_dict[4] = 18
saunders_crop_dict[7] = 22
saunders_crop_dict[11] = 28
saunders_crop_dict[18] = 38
saunders_crop_dict[20] = 47
saunders_crop_dict[30] = 61
saunders_crop_dict[35] = 70
saunders_crop_dict[38] = 75
saunders_crop_dict[39] = 76
saunders_crop_dict[40] = 77
saunders_crop_dict[46] = 84
saunders_crop_dict[54] = 120
saunders_crop_dict[57] = 125
saunders_crop_dict[59] = 132
saunders_crop_dict[67] = 9
saunders_crop_dict[69] = 12
saunders_crop_dict[70] = 14
saunders_crop_dict[71] = 15
saunders_crop_dict[72] = 16
saunders_crop_dict[73] = 17
saunders_crop_dict[86] = 110
saunders_crop_dict[94] = 118
saunders_crop_dict[97] = 128

# vandevelde (old -> new)
vandevelde_crop_dict = {}
vandevelde_crop_dict[0] = 0
vandevelde_crop_dict[2] = 10
vandevelde_crop_dict[4] = 13
vandevelde_crop_dict[7] = 33
vandevelde_crop_dict[8] = 35
vandevelde_crop_dict[9] = 37
vandevelde_crop_dict[10] = 40
vandevelde_crop_dict[12] = 5 
vandevelde_crop_dict[13] = 6
vandevelde_crop_dict[16] = 15
vandevelde_crop_dict[17] = 16
vandevelde_crop_dict[20] = 21
vandevelde_crop_dict[23] = 24
vandevelde_crop_dict[24] = 25
vandevelde_crop_dict[27] = 30
vandevelde_crop_dict[28] = 31
vandevelde_crop_dict[34] = 44
vandevelde_crop_dict[37] = 47
vandevelde_crop_dict[48] = 58
vandevelde_crop_dict[49] = 59
vandevelde_crop_dict[51] = 61
vandevelde_crop_dict[60] = 70
vandevelde_crop_dict[61] = 71
vandevelde_crop_dict[62] = 72

# global nested dict
id_updater_dict = {}
id_updater_dict['kiepert_1845'] = kiepert_crop_dict
id_updater_dict['saunders_1874'] = saunders_crop_dict
id_updater_dict['vandevelde_1846'] = vandevelde_crop_dict

def modify_font_labels(file_path):
    #print(file_path)
    for search_string in ['kiepert_1845', 'saunders_1874', 'vandevelde_1846']:
        if search_string in file_path:
            id_index = file_path.find(search_string) + len(search_string) + 1
            id_str = file_path[id_index:file_path.find('.jpeg')]
            if id_str.isdigit():
                new_id = id_updater_dict[search_string][int(id_str)]
                return file_path.replace(search_string + "_" + id_str, search_string + "_" + str(new_id))

fixed_ids = [[modify_font_labels(lbl[0]), modify_font_labels(lbl[1]), lbl[2]] for lbl in data_list]

json_dict = {}
json_dict['data'] = fixed_ids

with open('dependencies/ground_truth_labels/ground_truth_labels_fixed.json', 'w') as f:
    json.dump(json_dict, f)

import Evaluation
import ExtractHandling
import Grouping
import importlib
importlib.reload(Evaluation)
%load_ext autoreload
from PIL import Image
from itertools import combinations

group_padding = []
for map_name_in_strec in ["kiepert_1845", "saunders_1874", "vandevelde_1846"]:
    gt_labels_full = Evaluation.load_ground_truth_labels(map_name_in_strec, "components")
    gt_labels_full = ExtractHandling.cast_coords_as_Polygons(gt_labels_full)
    gt_labels_group = gt_labels_full[(gt_labels_full['multiline_g'].notnull()) & (gt_labels_full['multiline_g'] != "")]
    gt_labels_group['file_path'] = "dependencies/ground_truth_labels/cropped/" + map_name_in_strec + "_" + gt_labels_group.index.astype(str) + ".jpeg"
    for group_id in gt_labels_group['multiline_g'].value_counts().index:
        tmp = gt_labels_group[gt_labels_group['multiline_g'] == str(group_id)].copy()
        fps = tmp['file_path'].to_list()
        result = []
        for combo in combinations(fps, 2):
            result.append([combo[0], combo[1], 1])
        group_padding.extend(result)

def deduplicate_list_of_lists(lst):
    seen = set()
    result = []
    for sublist in lst:
        tuple_sublist = tuple(sublist)
        rever_sublist = (tuple_sublist[1], tuple_sublist[0], tuple_sublist[2])
        # If the tuple is not in the set, add it to the set and result list
        if tuple_sublist not in seen and rever_sublist not in seen:
            seen.add(tuple_sublist)
            seen.add(rever_sublist)
            result.append(sublist)
    return result

fixed_n_padded_ids = deduplicate_list_of_lists(fixed_ids + group_padding)

json_dict['data'] = fixed_n_padded_ids

with open('dependencies/ground_truth_labels/ground_truth_labels_fixed.json', 'w') as f:
    json.dump(json_dict, f)

