from shapely.geometry import Polygon, MultiPolygon
from itertools import combinations
import importlib
import Clustering
import TextRectify
import TextAmalgamate
import ExtractHandling
import SpotterWrapper
import Grouping
import BezierSplineMetric
import FontSimilarity
import numpy as np
import random
from collections import Counter
from itertools import combinations
from igraph import Graph

importlib.reload(SpotterWrapper)
importlib.reload(Grouping)
importlib.reload(Clustering)
importlib.reload(TextRectify)
importlib.reload(TextAmalgamate)
importlib.reload(ExtractHandling)
importlib.reload(BezierSplineMetric)
importlib.reload(FontSimilarity)

# draw edges between nodes where bezier and font values pass the threshold
def calculate_edges(g, font_threshold, bezier_threshold):
    node_indices = [i for i in range(len(g.vs))]
    all_pairwise_combs = list(combinations(node_indices, 2))
    for pair in all_pairwise_combs:
        i = pair[0]
        j = pair[1]
        node_i = g.vs[i]
        node_j = g.vs[j]
        if i != j and not g.are_connected(i, j) and (node_j['index'] in node_i['neighbours']):
            if g.vs[i]['font_similarities'][g.vs[j]['index']] > font_threshold and g.vs[i]['bezier_costs'][g.vs[j]['index']] < bezier_threshold:
                g.add_edge(i, j)
    return g

def remap_dictionary_of_indices(dict, indices_to_map):
    for from_num in indices_to_map:
        if from_num in dict.keys():
            to_num = indices_to_map[from_num]
            dict[to_num] = dict.pop(from_num)
    return dict

def remap_set_of_indices(set, indices_to_map):
    return {indices_to_map[element] if element in indices_to_map.keys() else element for element in set}

def update_ind_map(local_dict, global_dict, from_ind, to_ind):
    local_dict[from_ind] = to_ind
    global_dict[from_ind] = to_ind
    for key, value in global_dict.items():
        if value == from_ind:
            global_dict[key] = to_ind
    return local_dict, global_dict

# combine two labels
def combine_labels(v1, v2, global_indices_to_map):

    v1_ind = v1['index']
    v2_ind = v2['index']
    index_new = min(v1_ind, v2_ind)
    local_indices_to_map = {}
    if index_new == v1_ind:
        local_indices_to_map, global_indices_to_map = update_ind_map(local_indices_to_map, global_indices_to_map, v2_ind, index_new)
    else:
        local_indices_to_map, global_indices_to_map = update_ind_map(local_indices_to_map, global_indices_to_map, v1_ind, index_new)

    poly1 = v1['label'][0]
    poly2 = v2['label'][0]
    poly_new = poly1.union(poly2) # returns multipolygon object with disjoint polygons if polygons are disjoint

    text_list1 = v1['text_list']
    text_list2 = v2['text_list']
    text_list_new = text_list1 + text_list2

    label_new = (poly_new, '')

    bezier_costs1 = v1['bezier_costs']
    bezier_costs2 = v2['bezier_costs']
    bezier_costs_new = {key: min(bezier_costs1.get(key, float('inf')), bezier_costs2.get(key, float('inf'))) for key in set(bezier_costs1) | set(bezier_costs2)}
    bezier_costs_new = remap_dictionary_of_indices(bezier_costs_new, local_indices_to_map)

    font_similarities1 = v1['font_similarities']
    font_similarities2 = v2['font_similarities']
    font_similarities_new = {key: max(font_similarities1.get(key, 0), font_similarities2.get(key, 0)) for key in set(font_similarities1) | set(font_similarities2)}
    font_similarities_new = remap_dictionary_of_indices(font_similarities_new, local_indices_to_map)

    neighbours1 = set(v1['neighbours'])
    neighbours2 = set(v2['neighbours'])
    if neighbours1 is None:
        neighbours1 = []
    if neighbours2 is None:
        neighbours2 = []
    neighbours_new = neighbours1.union(neighbours2)
    neighbours_new.discard(v1_ind)
    neighbours_new.discard(v2_ind)
    neighbours_new = remap_set_of_indices(neighbours_new, local_indices_to_map)
    neighbours_new = list(neighbours_new)

    return index_new, label_new, bezier_costs_new, font_similarities_new, neighbours_new, text_list_new, global_indices_to_map

# combine two nodes by adding to new graph and using combine_labels() function above for attributes
def subgraph_contractor(subgraph, edges_calculated, font_threshold, bezier_threshold, global_indices_to_map):
    if edges_calculated:
        pass
    else:
        subgraph = calculate_edges(subgraph, font_threshold, bezier_threshold)
    edges = subgraph.get_edgelist()
    uncontracted_vertices = [i for i in range(len(subgraph.vs))]
    subgraph_new = Graph()
    for edge in edges:
        if edge[0] in uncontracted_vertices and edge[1] in uncontracted_vertices:
            index_new, label_new, bezier_costs_new, font_similarities_new, neighbours_new, text_list_new, global_indices_to_map = combine_labels(subgraph.vs[edge[0]], subgraph.vs[edge[1]], global_indices_to_map)
            subgraph_new.add_vertex(index = index_new, label = label_new, bezier_costs = bezier_costs_new, font_similarities = font_similarities_new, neighbours = neighbours_new, text_list = text_list_new)
            uncontracted_vertices.remove(edge[0])
            uncontracted_vertices.remove(edge[1])
    for vertex in uncontracted_vertices:
        tmp_v = subgraph.vs[vertex]
        subgraph_new.add_vertex(index = tmp_v['index'], label = tmp_v['label'], bezier_costs = tmp_v['bezier_costs'], font_similarities = tmp_v['font_similarities'], neighbours = tmp_v['neighbours'], text_list = tmp_v['text_list'])
    return subgraph_new, False, global_indices_to_map

# wrapper for continued combination until weak connected subgraph cannot be further contracted 
def subgraph_contractor_wrapper(subgraph, font_threshold, bezier_threshold, global_indices_to_map):
    edges_calculated = True
    base_len = 0
    contracted_len = len(subgraph.vs)
    while base_len != contracted_len:
        base_len = len(subgraph.vs)
        subgraph, edges_calculated, global_indices_to_map = subgraph_contractor(subgraph, edges_calculated, font_threshold, bezier_threshold, global_indices_to_map)
        contracted_len = len(subgraph.vs)
    return subgraph, global_indices_to_map

# prepare subgraphs for seq req base on font sim and bezier cost
def sl_seq_req(indices, labels, bezier_costs, font_similarities, neighbours, text_lists, font_threshold, bezier_threshold):
    print(str(len(labels)) + " labels.")
    # create graph from labels, extract weak connected components (for isolated seq req)
    label_dict_for_graph = dict(zip(['index','label', 'font_similarities', 'bezier_costs', 'neighbours', 'text_list'], [indices, labels, font_similarities, bezier_costs, neighbours, text_lists]))
    g = Graph()
    g.add_vertices(len(labels),attributes=label_dict_for_graph)
    g = calculate_edges(g, font_threshold, bezier_threshold)
    connected_subgraphs = g.decompose()
    global_indices_to_map = {}
    collapsed_subgraphs = []

    # seq req weak connected components
    for i, subgraph in enumerate(connected_subgraphs, start=1):
        if subgraph.vcount() > 1:
            subgraph, global_indices_to_map = subgraph_contractor_wrapper(subgraph, font_threshold, bezier_threshold, global_indices_to_map)
        collapsed_subgraphs.append(subgraph)

    iter_indices = []
    iter_labels = []
    iter_bezier_costs = []
    iter_font_similarities = []
    iter_neighbours = []
    iter_text_lists = []
    for i, subgraph in enumerate(collapsed_subgraphs, start=1):
        iter_indices.extend([node['index'] for node in subgraph.vs])
        iter_labels.extend([node['label'] for node in subgraph.vs])
        iter_bezier_costs.extend([remap_dictionary_of_indices(node['bezier_costs'], global_indices_to_map) for node in subgraph.vs])
        iter_font_similarities.extend([remap_dictionary_of_indices(node['font_similarities'], global_indices_to_map) for node in subgraph.vs])
        iter_neighbours.extend([remap_set_of_indices(node['neighbours'], global_indices_to_map) for node in subgraph.vs])
        iter_text_lists.extend([node['text_list'] for node in subgraph.vs])
    return iter_indices, iter_labels, iter_bezier_costs, iter_font_similarities, iter_neighbours, iter_text_lists

# wrapper for sl seq req (apply sl sr to weak connected subgraphs until no more connected components manifest) 
def sl_seq_req_wrapper(labels, bezier_costs, font_similarities, neighbours, font_threshold, bezier_threshold):
    xcens = [lab[0].centroid.x for lab in labels]
    ycens = [lab[0].centroid.y for lab in labels]
    texts = [lab[1] for lab in labels]
    text_lists = [[(np.array([xcens[i], ycens[i]]), texts[i])] for i in range(len(labels))]
    base_len = 0
    sreq_len = len(labels)
    indices = [i for i in range(len(labels))]
    while base_len != sreq_len:
        base_len = len(labels)
        indices, labels, bezier_costs, font_similarities, neighbours, text_lists = sl_seq_req(indices, labels, bezier_costs, font_similarities, neighbours, text_lists, font_threshold, bezier_threshold)
        sreq_len = len(labels)
    print("Retained " + str(len(labels)) + ".")
    sorted_text_lists = [sorted(lst, key=lambda x: x[0][0]) for lst in text_lists]
    new_texts = [' '.join([t[1] for t in inner_lst]) for inner_lst in sorted_text_lists]
    labels_w_annotation = [(x[0], new_texts[i]) for i, x in enumerate(labels)]
    return indices, labels_w_annotation, bezier_costs, font_similarities, neighbours, text_lists