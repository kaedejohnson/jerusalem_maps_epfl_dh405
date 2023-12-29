import numpy as np
from shapely.geometry import Polygon, MultiPolygon
from collections import Counter
from itertools import combinations
from igraph import Graph

# iom of either text or polygon/multipolygon
def intersection_over_minimum(obj1, obj2):
    if (isinstance(obj1, Polygon) or isinstance(obj1, MultiPolygon)) and (isinstance(obj2, Polygon) or isinstance(obj2, MultiPolygon)):
        IoM = obj1.intersection(obj2).area / min(obj1.area, obj2.area)
    elif isinstance(obj1, str) and isinstance(obj2, str):
        obj1 = obj1.lower()
        obj2 = obj2.lower()
        cntr1 = Counter(obj1)
        cntr2 = Counter(obj2)
        global_char_set = set(cntr1.keys()) | set(cntr2.keys())
        IoM = sum(min(cntr1[char], cntr2[char]) for char in global_char_set) / min(len(obj1), len(obj2))
    else:
        print(obj1, obj2)
        print("both inputs must be of the same type (Polygon or string)")
        IoM = np.nan
    return IoM

# wrapper for iom function
def IoMs(label1, label2):
    poly1 = label1[0]
    text1 = label1[1]
    poly2 = label2[0]
    text2 = label2[1]
    if len(set(text1.lower()) | set(text2.lower())) == 0:
        return (0, 0)
    if not poly1.intersects(poly2):
        return (0, 0)
    poly_IoU = intersection_over_minimum(poly1, poly2)
    text_IoU = intersection_over_minimum(text1, text2)
    return (poly_IoU, text_IoU)

# draw edges between nodes where node 'label' attribute has sufficient IoMs
def calculate_edges(g, geo_threshold, text_threshold):
    node_indices = range(len(g.vs))
    all_pairwise_combs = list(combinations(node_indices, 2))
    for pair in all_pairwise_combs:
        i = pair[0]
        j = pair[1]
        if i != j and not g.are_connected(i, j):
            node1_label = g.vs[i]['label']
            node2_label = g.vs[j]['label']
            weight = IoMs(node1_label, node2_label)
            if weight[0] > geo_threshold and weight[1] > text_threshold:
                g.add_edge(i, j, weight=weight)
    return g

# combine two labels
def combine_labels(label1, label2):
    poly1 = label1['label'][0]
    text1 = label1['label'][1]
    poly2 = label2['label'][0]
    text2 = label2['label'][1]
    poly_new = poly1.union(poly2)
    text_new = text1 if len(text1) > len(text2) else text2
    return (poly_new, text_new)

# combine two nodes by adding to new graph and using combine_labels() function above for attribute
def subgraph_contractor(subgraph, edges_calculated, geo_threshold, text_threshold):
    if edges_calculated:
        pass
    else:
        subgraph = calculate_edges(subgraph, geo_threshold, text_threshold)
    edges = subgraph.get_edgelist()
    uncontracted_vertices = {i for i in range(len(subgraph.vs))}
    subgraph_new = Graph()
    for edge in edges:
        if edge[0] in uncontracted_vertices and edge[1] in uncontracted_vertices:
            subgraph_new.add_vertex(label = combine_labels(subgraph.vs[edge[0]], subgraph.vs[edge[1]]))
            uncontracted_vertices.remove(edge[0])
            uncontracted_vertices.remove(edge[1])
    for vertex in uncontracted_vertices:
        subgraph_new.add_vertex(label = subgraph.vs[vertex]['label'])
    return subgraph_new, False

# wrapper for continued combination until weak connected subgraph cannot be further contracted 
def subgraph_contractor_wrapper(subgraph, geo_threshold, text_threshold):
    edges_calculated = True
    base_len = 0
    contracted_len = len(subgraph.vs)
    while base_len != contracted_len:
        base_len = len(subgraph.vs)
        subgraph, edges_calculated = subgraph_contractor(subgraph, edges_calculated, geo_threshold, text_threshold)
        contracted_len = len(subgraph.vs)
    return subgraph

# prepare subgraphs for flattening base on IoMs
def nwf(labels, geo_threshold, text_threshold):

    # create graph from labels, extract weak connected components (for isolated flattening)
    label_dict_for_graph = dict(zip(['label'], [labels]))
    g = Graph()
    g.add_vertices(len(labels),attributes=label_dict_for_graph)
    g = calculate_edges(g, geo_threshold, text_threshold)
    connected_subgraphs = g.decompose()

    # flatten weak connected components
    nwf_iteration = []
    for i, subgraph in enumerate(connected_subgraphs, start=1):
        if subgraph.vcount() > 1:
            subgraph = subgraph_contractor_wrapper(subgraph, geo_threshold, text_threshold)
        nwf_iteration.extend([node['label'] for node in subgraph.vs])

    return nwf_iteration

# wrapper for nwf (apply nwf to weak connected subgraphs until no more connected components manifest) 
def nwf_wrapper(labels, geo_threshold, text_threshold):
    print("Started NWF with " + str(len(labels)) + " labels.")
    base_len = 0
    flattened_len = len(labels)
    while base_len != flattened_len:
        base_len = len(labels)
        labels = nwf(labels, geo_threshold, text_threshold)
        flattened_len = len(labels)
    print("Retained " + str(len(labels)) + ".")
    return labels