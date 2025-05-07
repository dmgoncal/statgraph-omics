import numpy as np
import networkx as nx
import itertools
import scipy
import operator
from sklearn.neighbors import KernelDensity


compare_functions = {"ks": lambda x,y: scipy.stats.ks_2samp(x, y).pvalue, 
                     "ks_stat": lambda x,y: scipy.stats.ks_2samp(x, y).statistic,
                     "kl_div": lambda x,y: np.sum(scipy.special.kl_div(x,y))}
compare_functions_best = {'ks': min,
                          'ks_stat': max}

comparison_functions = {"lower": operator.lt, "higher": operator.gt, "lower or equal": operator.le, 
                        "higher or equal": operator.ge, "equal": operator.eq, "not equal": operator.ne}

# create distribution relating two distributions
def get_ratio_distribution(x1, x2, delta = 0.00001):
    x1, x2 = np.array(x1) + delta, np.array(x2) + delta
    ratio_dist = x1/x2
    return ratio_dist

def get_log_ratio_distribution(x1, x2):
    ratio_dist = get_ratio_distribution(x1, x2)
    log_ratio_dist = np.log(ratio_dist)
    return log_ratio_dist

def get_abs_diff_distribution(x1, x2):
    x1, x2 = np.array(x1), np.array(x2)
    diff_dist = np.abs(x1 - x2)
    return diff_dist

combine_functions = {"ratio": get_ratio_distribution, 
                     "diff": get_abs_diff_distribution, 
                     "log_ratio": get_log_ratio_distribution}


def attribute_mean(g, where, attribute, weighted_by=None):
    if where.lower() == 'node':
        graph_el = g.nodes
    elif where.lower() == 'edge':
        graph_el = g.edges
    else:
        raise ValueError(f"Parameter {{to_filter}} must be one of the following: node or edge")
    
    value_list = []
    for el in graph_el:
        if weighted_by is None:
            value = graph_el[el][attribute]
        else:
            value = graph_el[el][attribute] * graph_el[el][weighted_by]
        value_list.append(value)

    res = np.mean(value_list)
    return res


def classify_sample_graph(g, classification, weight_attr=None, verbose=0):
    if nx.is_empty(g):
        raise RuntimeError('Graph is empty')
    classif_values = ['edge', 'edges', 'node', 'nodes', 'both']
    if classification.lower() not in classif_values:
        raise ValueError(f"Parameter {{classification}} cannot be '{classification}', must be one of the following: {classif_values}")

    edge_attributes = list(g.edges(data=True))[0][2]
    class_attributes = [at for at in edge_attributes if 'likelihood' in at]

    if classification.lower() in ['edge', 'edges', 'both']:
        edge_means_dict = {}
        for c_attr in class_attributes:
            edge_mean = attribute_mean(g, 'edge', c_attr, weight_attr)
            edge_means_dict[c_attr] = edge_mean
        final_results = edge_means_dict.copy()


    if classification.lower() in ['node', 'nodes', 'both']:
        node_means_dict = {}
        for c_attr in class_attributes:
            node_mean = attribute_mean(g, 'node', c_attr, weight_attr)
            node_means_dict[c_attr] = node_mean
        final_results = node_means_dict.copy()

    if classification.lower() == 'both':
        sum_edge = abs(sum(edge_means_dict.values()))
        sum_node = abs(sum(node_means_dict.values()))
        final_results = {}
        for c_attr in class_attributes:
            edge_likelihood = edge_means_dict[c_attr]/sum_edge if sum_edge > 0 else 0 # may require revision
            node_likelihood = node_means_dict[c_attr]/sum_node if sum_edge > 0 else 0
            final_results[c_attr] = edge_likelihood + node_likelihood

    likelihoods = {key: np.exp(value) for key, value in final_results.items()}
    total_likelihood = sum(likelihoods.values())
    probabilities = {key: likelihood / total_likelihood for key, likelihood in likelihoods.items()}
    sorted_classes = sorted(int(key.split('_')[0]) for key in probabilities.keys())
    probabilities = [probabilities[f'{cls}_likelihood'] for cls in sorted_classes]

    prediction = max(final_results, key=final_results.get)
    prediction = prediction.replace('_likelihood', '')
    return prediction, probabilities


# CREATION OF SAMPLE GRAPHS EDGE BY EDGE FOR MEMORY EFFICIENCY

def pair_combine_class_distributions(x_data, y_data, var_pair, combine_distributions, verbose=0):
    if len(var_pair) != 2:
        raise ValueError(f"Paramater {{var_pair}} should have two elements, not {len(var_pair)} as in {var_pair}.")
    
    try:
        combine_function = combine_functions[combine_distributions]
    except KeyError:
        raise ValueError(f"Parameter {{combine_distributions}} cannot be '{combine_distributions}', must be one of the following: {list(combine_functions.keys())}")
    
    var1, var2 = sorted(var_pair)[0], sorted(var_pair)[1] 

    x1, x2 = x_data[var1].values, x_data[var2].values
    classes = np.unique(y_data)

    distributions = {}
    for c in classes:
        x1_c = x1[y_data == c]
        x2_c = x2[y_data == c]
        distributions[c] = (x1_c, x2_c)

    combined_distributions = {}
    for c, dist in distributions.items():
        combined_distributions[c] = combine_function(dist[0], dist[1]) 

    return combined_distributions


def pair_sample_combine_value(x_data, var_pair, combine_distributions, verbose=0):
    try:
        combine_function = combine_functions[combine_distributions]
    except KeyError:
        raise ValueError(f"Parameter {{combine_distributions}} cannot be '{combine_distributions}', must be one of the following: {list(combine_functions.keys())}")
    
    var1, var2 = sorted(var_pair)[0], sorted(var_pair)[1] 
    x1, x2 = x_data[var1], x_data[var2]
    new_val = combine_function(x1, x2)

    return new_val


def pair_difference_score(distributions, compare_distributions):
    try:
        compare_function = compare_functions[compare_distributions]
    except KeyError:
        raise ValueError(f"Parameter {{compare_distributions}} cannot be '{compare_distributions}', must be one of the following: {list(compare_functions.keys())}")
    
    best_function = compare_functions_best[compare_distributions]
    
    scores_list = []
    classes = list(distributions.keys())
    for class_pair in itertools.combinations(classes, 2):
        dist1, dist2 = distributions[class_pair[0]], distributions[class_pair[1]]
        score = compare_function(dist1, dist2)
        scores_list.append(score)
    return best_function(scores_list)


def pair_classification_score(value, distributions: dict, verbose=0):
    
    likelihoods = {}
    classes = list(distributions.keys())
    for c in classes: 
        distribution = distributions[c]
        kernel = KernelDensity(kernel='gaussian').fit(distribution.reshape(-1, 1))
        likelihood = kernel.score(value.reshape(1,1))
        key = f'{c}_likelihood'
        likelihoods[key] = likelihood
    return likelihoods

def create_sample_graphs_from_edges(dict_data: dict, weights: dict = None, diff_scores = None):
    graphs = {}

    for sample, data in dict_data.items():
        sample_graph = nx.Graph()
        edges = []
        for var_pair, attr in data.items(): 
            if weights is not None:
                attr['weight'] = weights[var_pair]
            if diff_scores is not None:
                attr['diff_score'] = diff_scores[var_pair]
            edges.append([var_pair[0], var_pair[1], attr])

        sample_graph.add_edges_from(edges)  
        graphs[sample] = sample_graph

    return graphs


def add_node_data_to_samples(samples: dict, dict_data: dict, weights: dict = None):
    for sample, graph in samples.items():
        nodes = []
        for node, attr in dict_data[sample].items():
            if weights is not None:
                attr['weight'] = weights[node]
            nodes.append((node, attr))
        
        graph.add_nodes_from(nodes)
    return samples


def transform_weights(samples: dict, where: str, inverse: bool, normalize: bool, minmax: bool, 
                      mymin = 0, mymax = 1, weight_attr = 'weight', verbose=0):
    if not inverse and not normalize and not minmax:
        print(f'Warning: If {{inverse}} and {{normalize}}  and {{minmax}} are False, then there is nothing to do...')
        return samples

    if verbose>0:
        g = samples[list(samples.keys())[0]]
        if where.lower() in ['node', 'nodes']:
            print("Before: example node weight:", g.nodes[list(g.nodes)[0]][weight_attr])
        elif where.lower() in ['edge', 'edges']:
            print("Before: example edge weight: ", g.edges[list(g.edges)[0]][weight_attr])

    # transform weights of all graphs 
    for sample_id, graph in samples.items():

            if where.lower() in ['node', 'nodes']:
                graph_elements = graph.nodes
            elif where.lower() in ['edge', 'edges']:
                graph_elements = graph.edges
            else:
                raise ValueError(f"Parameter {{where}} must be one of the following: node or edge")
            
            weights_sum = 0
            maxv, minv = None, None
            for el in graph_elements:
                weight = graph_elements[el][weight_attr]
                weight = 1/weight if inverse else weight
                graph_elements[el][weight_attr] = weight
                weights_sum += weight
                if maxv is None or weight>maxv:
                    maxv = weight
                if minv is None or weight<minv:
                    minv = weight

            if normalize:
                for el in graph_elements:
                    graph_elements[el][weight_attr] = graph_elements[el][weight_attr]/weights_sum     

            if minmax:
                epsilon = 1e-10
                for el in graph_elements:
                    graph_elements[el][weight_attr] = ((graph_elements[el][weight_attr] - minv) / (maxv - minv + epsilon)) * (mymax - mymin) + mymin

    if verbose>0:
        g = samples[list(samples.keys())[0]]
        if where.lower() in ['node', 'nodes']:
            print("After: example node weight:", g.nodes[list(g.nodes)[0]][weight_attr])
        elif where.lower() in ['edge', 'edges']:
            print("After: example edge weight: ", g.edges[list(g.edges)[0]][weight_attr])

    return samples
