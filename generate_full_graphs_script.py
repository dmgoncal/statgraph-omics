import argparse
import itertools
from multiprocessing import Pool
import os
import pandas as pd
from tqdm import tqdm
import networkx as nx
import numpy as np
import graph_creation as gc
from functions import get_data
from constants import CV_RESULTS_FILE, DATA_DIR


DISABLE = False

def create_full_graph_edges(x_data, y_data, pvalue: float, features: list[str], 
                            processes: int|float, disable=False, verbose=0):
    if verbose>0: print('Starting create_full_graph_edges')
    edges = []
    total_number_edges = len(list(itertools.combinations(features, 2)))
    
    args = ((e, x_data[list(e)], y_data, pvalue) for e in itertools.combinations(features, 2))
    with Pool(processes=processes) as pool:
        # Create iterator for multiprocessing with function and arguments for each call
        results_iterator = pool.imap_unordered(create_edge_wrapper, args)
        # Iterate over iterator
        for edge in tqdm(results_iterator, total=total_number_edges, disable=disable):
            if edge is not None:
                edges.append(edge)      

    full_graph = nx.Graph()
    full_graph.add_edges_from(edges)
    return full_graph


def create_edge(edge: tuple[str], x_data, y_data, pvalue: float):
    # get class distributions
    distributions = gc.pair_combine_class_distributions(x_data, y_data, edge, 'log_ratio')
    # get difference score for class distributions
    diff_score = gc.pair_difference_score(distributions, compare_distributions='ks')
    weight = gc.pair_difference_score(distributions, compare_distributions='ks_stat')

    # filter out edges that do not reach the defined threshold
    comparison_function = gc.comparison_functions['lower']
    if not comparison_function(diff_score, pvalue):
        return None
    else:
        node1, node2 = sorted(edge)[0], sorted(edge)[1] 
        return (node1, node2, {'p-value': diff_score, 'weight': weight})

def create_edge_wrapper(args):
    return create_edge(*args)


def create_full_graph_add_nodes(x_data, y_data, graph,
                                processes: int|float, disable=False, verbose=0):
    
    args = ((n, x_data[[n]], y_data,) for n in graph.nodes)
    with Pool(processes=processes) as pool:
        # Create iterator for multiprocessing with function and arguments for each call
        results_iterator = pool.imap_unordered(add_node_data_wrapper, args)
        # Iterate over iterator
        for node, weight in tqdm(results_iterator, total=len(graph.nodes), disable=disable):
            graph.nodes[node]['weight'] = weight
    return graph


def add_node_data(node, x_data, y_data):
    # get class distributions
    dist = x_data[node].values
    classes = np.unique(y_data)
    distributions = {}
    for c in classes:
        x_c = dist[y_data == c]
        distributions[c] = x_c
    # get difference score for class distributions
    weight = gc.pair_difference_score(distributions, compare_distributions='ks_stat')
    return node, weight

def add_node_data_wrapper(args):
    return add_node_data(*args)


def save_graph(g, project, layer, target):
    file = f"./graphs/{project}_{target}_{layer}.gexf"
    exists = os.path.isfile(file)
    print(g)
    nx.write_gexf(g, file)


def main():
    ### Parse arguments ########################################################################################
    parser = argparse.ArgumentParser(description="Run graph classification in choosen TCGA data and store results")
    parser.add_argument("project", help="TCGA project to use", type=str.upper)
    parser.add_argument("layer", help="Layer of omics to use", type=str.lower, choices=['mirna', 'mrna', 'protein'])
    parser.add_argument("target", help="Variable to set as target", type=str.lower, choices=['vital_status', 'primary_diagnosis'])
    parser.add_argument("processes", help="Number (int) or fraction (float) of processors to use")
    # Flags
    parser.add_argument("-v", "--verbosity", action="count", default=0, help="Increase output verbosity")
    parser.add_argument("-r", "--reduced", type=int, help="Reduce the number of features to the specified number") 
    parser.add_argument("-s", "--skip", action="store_true", help="Skip already calculated results") 
    parser.add_argument("-y", "--yes", action="store_true", help="Recalculate results") 

    args = parser.parse_args()

    # Atribute args to variables
    project = args.project
    layer = args.layer
    target = args.target
    processes = eval(args.processes)
    # Flags
    reduced = args.reduced
    verbose = args.verbosity
    skip = args.skip
    yes = args.yes
    processes = int(processes*os.cpu_count()) if isinstance(processes, float) else processes

    # Get best results and corresponding parameterizations
    cv_results = pd.read_csv(CV_RESULTS_FILE)
    # Get row  corresponding to desired project, layer, and target
    row = cv_results[(cv_results['project']==project) & (cv_results['layer']==layer) & (cv_results['target']==target)]
    # Get corresponding parameterizations
    row = row.replace({np.nan: None})
    pvalue, weighted, minmax, dge = row['pvalue'].values[0], row['weighted'].values[0], row['minmax'].values[0], eval(str(row['dge_filtering'].values[0]))

    # Get data
    data, features = get_data(DATA_DIR, project, layer, target, dge, reduced, verbose)
    x, y = data[features], data[target]

    # Create graph
    graph = create_full_graph_edges(x, y, pvalue, features, processes, DISABLE, verbose)
    # Add node data
    graph = create_full_graph_add_nodes(x, y, graph, processes, DISABLE, verbose)
    # Save graph
    save_graph(graph, project, layer, target)


if __name__ == "__main__":
    main()
