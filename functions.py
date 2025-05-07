import itertools
from multiprocessing import Pool
import sys
import graph_creation as gc
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, roc_auc_score
from pydge.pydge import deg_filtering
from sklearn.model_selection import StratifiedKFold
from tcgahandler import LayerDataset
from constants import CV_FOLD_RESULTS_FILE, ML_RESULTS_FILE_CV_FOLD
from scipy.stats import ttest_rel


comparison = "lower"
combination = 'log_ratio'

###########################################################################################
########### GENERAL FUNCTIONS #######################################################
###########################################################################################

def postprocess_cv_avg(output_filepath : str, group_keys : list = None):
    """
    Processes cross-validation results to find the average performance over the folds.
    """
    df = pd.read_csv(CV_FOLD_RESULTS_FILE)

    # Define group keys if not provided
    group_keys = [
        "project", "layer", "target", "pvalue", "classification_type",
        "weighted", "minmax", "dge_filtering"
    ] if group_keys is None else group_keys

    # Fill NaN values with 0.0 for numeric columns and 'N/A' for group keys
    df = df.fillna({n: 0.0 if n not in group_keys else 'N/A' for n in df.columns})

    # Separate group keys and metrics
    metrics = [col for col in df.columns if col not in group_keys + ['fold']]
    
    # Compute mean and standard deviation
    agg_funcs = {metric: ['mean', 'std'] for metric in metrics}
    grouped = df.groupby(by=group_keys).agg(agg_funcs)

    # Flatten multi-level columns resulting from multiple aggregation functions
    grouped.columns = ['_'.join(col).strip() if col[1]=='std' else col[0] for col in grouped.columns.values]
    
    # Reset index and save to CSV
    grouped = grouped.reset_index()
    grouped.to_csv(output_filepath, index=False)


def postprocess_cv_best(output_filepath : str, decision_cols : list = None, group_keys : list = None):
    """
    Processes cross-validation results to find the best parameter sets.
    """
    df = pd.read_csv(CV_FOLD_RESULTS_FILE)
    # In order of relevance
    decision_cols = ['f1_score','recall','precision','accuracy'
                    ] if decision_cols is None else decision_cols
    group_keys = ["project","layer","target","pvalue","classification_type",
                  "weighted","minmax","dge_filtering"
                    ] if group_keys is None else group_keys
    df = df.fillna({n: 0.0 if n not in group_keys else 'N/A' for n in df.columns})
    
    metrics = [col for col in df.columns if col not in group_keys + ['fold']]
    # Compute mean and standard deviation
    agg_funcs = {metric: ['mean', 'std'] for metric in metrics}
    grouped = df.groupby(by=group_keys).agg(agg_funcs)
    # Flatten multi-level columns resulting from multiple aggregation functions
    grouped.columns = ['_'.join(col).strip() if col[1]=='std' else col[0] for col in grouped.columns.values]
    df = grouped.reset_index()

    # Filter best parameter sets
    for c in decision_cols:
        df = df.loc[df.reset_index().groupby(['project','layer','target'])[c].idxmax()]
        if not df.duplicated(subset=['project','layer','target']).any():
            break
    # Remove the fold column and rebuild the index before writing
    df = df.loc[:, df.columns != 'fold']
    df = df.reset_index()
    df.iloc[:,1:].to_csv(output_filepath, index=False)


def postprocess_graph_vs_ml(output_filepath : str):
    """
    Processes cross-validation results to calculate t-test stats between graph vs ML baselines.
    """
    df_graph = pd.read_csv(CV_FOLD_RESULTS_FILE)
    df_ml = pd.read_csv(ML_RESULTS_FILE_CV_FOLD)

    # Define group keys if not provided
    group_keys_graph = ["project", "layer", "target", "pvalue", "classification_type","weighted", "minmax", "dge_filtering"]
    group_keys_ml = ["project","layer","target","classifier","dge_filtering"]

    # Fill NaN values with 0.0 for numeric columns and 'N/A' for group keys
    df_graph = df_graph.fillna({n: 0.0 if n not in group_keys_graph else 'N/A' for n in df_graph.columns})
    df_ml = df_ml.fillna({n: 0.0 if n not in group_keys_ml else 'N/A' for n in df_ml.columns})

    # Separate group keys and metrics
    metrics = ['accuracy','f1_score','recall','precision','auc']
    
    # Compute mean and standard deviation
    grouped_graph = df_graph.groupby(by=group_keys_graph)
    grouped_ml = df_ml.groupby(by=group_keys_ml)
    
    # Initialize results list
    results = []

    # Iterate over metrics
    for metric in metrics:
        # Iterate over unique groupings
        for graph_key, graph_group in grouped_graph:
            for ml_key, ml_group in grouped_ml:
                # Ensure groupings match on common fields
                if all(graph_key[i] == ml_key[i] for i in range(3)):
                    # Extract metric values
                    graph_values = graph_group[metric].values
                    ml_values = ml_group[metric].values
                    
                    # Ensure the lengths of values are the same (paired comparison)
                    if len(graph_values) == len(ml_values):
                        # Perform paired t-test
                        t_stat, p_value = ttest_rel(graph_values, ml_values)
                        
                        # Append results
                        results.append({
                            **dict(zip(group_keys_graph, graph_key)),
                            'ml_model': list(ml_group['classifier'])[0],
                            'metric': metric,
                            'mean_graph': graph_values.mean(),
                            'std_graph': graph_values.std(),
                            'mean_ml': ml_values.mean(),
                            'std_ml': ml_values.std(),
                            't-statistic': t_stat,
                            'p-value': p_value
                        })

    # Convert results to DataFrame
    results_df = pd.DataFrame(results)

    # Save results to CSV
    results_df.to_csv(output_filepath, index=False)


def save_results(results: dict, file: str, drop_duplicates_subset: list[str]):
    """
    Save row of results in existing dataframe or creates a new one
        'drop_duplicates_subset' represent the columns to compare to find duplicates
    """
    try:
        df = pd.read_csv(file)
    except FileNotFoundError:
        df = pd.DataFrame(columns=list(results.keys()))
    # Add results
    df = pd.concat([df, pd.DataFrame([results])], ignore_index=True)
    # Remove replicate results, keep the last occurence
    df = df.drop_duplicates(subset=drop_duplicates_subset, keep='last')
    # Save to file
    df.to_csv(file, index=False)


def confirm_result_exists(row_values: dict, data_filename: str) -> str:
    '''
    Check if a given result exists in a given csv file
    '''
    try:
        df = pd.read_csv(data_filename)
        for k,v in row_values.items():
            df = df[df[k]==v]
            if df.empty:
                return False
            
        return True
    except FileNotFoundError:
        return False


def check_result_exists(results_to_check: dict, file: str, skip_flag: bool, yes_flag: bool):
    '''
    Check if result is already saved and exit according to flags
    '''
    exists = confirm_result_exists(results_to_check, file)
    if exists and skip_flag:
        print("Result already calculated, skiping...")
        sys.exit()
    elif exists and yes_flag:
        print("Recalculating results...")
    elif exists and not skip_flag:
        input_skip = input("Result already exists, want to recalculate?\t")
        if input_skip.lower() != 'y':
            sys.exit()


def check_ml_result_exists(classifiers: list[str], results_to_check: dict, file: str,
                           skip_flag: bool, yes_flag: bool):
    '''Check if result already exists and exit according to flags'''
    classifiers_to_use = {}
    for classif in classifiers.keys():
        results_to_check['classifier'] = classif
        exists = confirm_result_exists(results_to_check, file)
        if exists and skip_flag:
            print(f"Result already calculated for {classif}, skiping...")
        elif exists and yes_flag:
            classifiers_to_use[classif] = classifiers[classif]
            print(f"Recalculating results for {classif}...")
        elif exists and not skip_flag:
            input_skip = input(f"Result already exists for {classif}, want to recalculate?\t")
            if input_skip.lower() == 'y':
                classifiers_to_use[classif] = classifiers[classif]
        else:
            print(f'Calculating for {classif}')
            classifiers_to_use[classif] = classifiers[classif]

    if len(classifiers_to_use)==0:
        print('No classifiers need calculation, exiting...')
        sys.exit()
    else:
        return classifiers_to_use


###########################################################################################
########### DATA FUNCTIONS ################################################################
###########################################################################################
            
def get_data(data_dir: str, project: str, layer: str, target: str, dge: int|float, 
             reduced: int = None, verbose=0):
    '''
    Get corresponding dataset and preprocess it according to flags
    '''
    def dge_preprocess(data):
        '''Perform DGE Analysis and filter according to it'''
        if dge is not None and layer=='protein':  # DGE filtering
            print('DGE filtering not implemented for protein layer. Continuing...')
        elif dge is not None:
            counts = dataset.get_data_with_target(data_type='counts', target=target)
            if dge < 1: # filter by p-value
                print('pvalue')
                filtered_genes = deg_filtering(counts, target, pvalue=dge)
            elif dge > 1: # select top n genes
                print('n genes')
                filtered_genes = deg_filtering(counts, target, n_genes=int(dge))
            else:
                raise ValueError('Parameter --dge must be int or float, not:', dge, type(dge))
            del counts
            data = data[filtered_genes + [target]]
        return data
    
    def minmax_normalization(data, features):
        minimum, maximum = data[features].to_numpy().min(), data[features].to_numpy().max()
        data[features] = (data[features] - minimum) / (maximum - minimum)
        return data

    dataset = LayerDataset(data_dir, project, layer)
    data = dataset.get_data_with_target('default', target, log_transform=False)
    # for testing purposes
    if reduced is not None:
        target_col = data[target]
        data = data.iloc[:,:reduced] 
        data[target] = target_col
    # DGE filtering
    if verbose>0: print(f'Data shape before DGE filtering: {data.shape}')
    data = dge_preprocess(data)
    if verbose>0: print(f"Data shape after DGE filtering: {data.shape}")
    # save feature names
    features = data.drop(columns=[target]).columns.tolist()
    # minmax normalization
    data = minmax_normalization(data, features)
    return data, features


###########################################################################################
########### GRAPH CREATION FUNCTIONS ######################################################
###########################################################################################

def prune_sample_graphs(sample_graphs: dict, pvalue):
    """
    Receives  multiple sample graphs and filters out edges that do not meet the threshold {pvalue}
    """
    graph_ex = list(sample_graphs.values())[0]
    edges_to_remove = [(a,b) for a, b, attrs in graph_ex.edges(data=True) if attrs["diff_score"] >= pvalue]
    for id, graph in sample_graphs.items():
        graph.remove_edges_from(edges_to_remove)
        sample_graphs[id] = graph
    return sample_graphs


###########
#  EDGES  #
###########

def parallel_graph_sample_creation(edge, x_train, y_train, x_test, pvalue):
    '''
    Computes edge-based classification scores for test samples using KS-based class distribution differences.
    '''
    # initialize dict that will save data about a given edge across samples as: 
    #   {'sample_id': {'attr1': value, 'attr2': value}, 'sample_id2': ...}
    edges_dict = {}

    # get class distributions
    distributions = gc.pair_combine_class_distributions(x_train, y_train, edge, combination)

    # get difference score for class distributions
    diff_score = gc.pair_difference_score(distributions, compare_distributions='ks') # p-value
    weight = gc.pair_difference_score(distributions, compare_distributions='ks_stat') # ks weight

    # filter out edges that do not reach the defined threshold (pvalue)
    comparison_function = gc.comparison_functions[comparison]
    if not comparison_function(diff_score, pvalue):
        return (edge, {}, None, None)

    # go through all test samples and assign classification scores for the given edge
    for ind in x_test.index:
        sample = x_test.loc[ind, list(edge)]
        # this is a single value resulting from applying the combination function on two values
        sample_value = gc.pair_sample_combine_value(sample, edge, combination)
        # this is of type: {'pos_likelihood': -1.234, 'neg_likelihood': -4.345}
        classification_scores = gc.pair_classification_score(sample_value, distributions)
        # reduce from float64 to float 32
        classification_scores = {k: np.float32(v) for k,v in classification_scores.items()}
        # add to dictionary that will be used to create graphs
        edges_dict[ind] = classification_scores
    # reduce from float64 to float 32
    weight = np.float32(weight)
    diff_score = np.float32(diff_score)

    return (edge, edges_dict, weight, diff_score)

def parallel_graph_sample_creation_wrapper(args):
    return parallel_graph_sample_creation(*args)


def sample_graph_edges_creation(x_train, y_train, x_test, pvalue, features, processes, 
                                disable=False, verbose=0):
    '''
    Constructs sample-specific graphs by computing classification scores for all feature pairs in parallel.
    '''
    if verbose>0: print('Starting sample graph edge creation')
    total_number_edges = len(list(itertools.combinations(features, 2)))
    # Initialize dict to hold information about all edges, will be of type:
    #   {'sample_id1': {('node1', 'node2'): {'pos_likelihood': -1.58, 'neg_likelihood': -1.86}, 
    #   ('node2', 'node3'): {'pos_likelihood': -1.31, 'neg_likelihood': -1.16}}, 
    #   'sample_id2': {('node1', 'node2'): ...}
    sample_edges_dict = {}
    for ind in x_test.index:
        sample_edges_dict[ind] = {}
    sample_edges_weights = {}
    sample_edges_diff = {}
    # Generator with parameters for parallel function. Of type:
    #   ( (edge1, [20,31,45...], [0, 0, 1...], [13, 26, 43...], 0.001), .... )
    parallel_args = ((e, x_train[list[e]], y_train, x_test[list[e]], pvalue) for e in itertools.combinations(features, 2))
    # Create pool of processes for parallelization
    with Pool(processes=processes) as pool:
        # Create iterator for multiprocessing with function and arguments for each call
        results_iterator = pool.imap_unordered(parallel_graph_sample_creation_wrapper, parallel_args)
        # Iterate over iterator
        for result in tqdm(results_iterator, total=total_number_edges, disable=disable):
            edge, edges_dict, weight, diff_score = result
            if weight is not None:
                sample_edges_weights[edge] = weight
            if diff_score is not None:
                sample_edges_diff[edge] = diff_score
            for k,v in edges_dict.items():
                    sample_edges_dict[k][edge] = v
    sample_graphs = gc.create_sample_graphs_from_edges(sample_edges_dict, sample_edges_weights, sample_edges_diff)
    if verbose>0: print('Sample graph edge creation finished')
    return sample_graphs


###########
#  NODES  #
###########

def parallel_get_node_data(node, x_train, y_train, x_test):
    '''
    Computes per-sample classification scores and KS-stat difference for a single node across classes.
    '''
    # initialize dict that will save data about a given node across samples as: 
    # {'sample_id': {'attr1': value, 'attr2': value}, 'sample_id2': ...}
    nodes_dict = {}

    # get class distributions
    dist = x_train[node].values
    classes = np.unique(y_train)
    distributions = {}
    for c in classes:
        x_c = dist[y_train == c]
        distributions[c] = x_c
    # get difference score for class distributions
    diff_score = gc.pair_difference_score(distributions, compare_distributions='ks_stat')

    # go through all test samples and assign classification scores
    for ind in x_test.index:
        sample_value = x_test.loc[ind, node]
        classification_scores = gc.pair_classification_score(sample_value, distributions)
        # reduce from float64 to float 32
        classification_scores = {k: np.float32(v) for k,v in classification_scores.items()}
        # add to dictionary that will be used to create graphs
        nodes_dict[ind] = classification_scores
    
    return (node, nodes_dict, diff_score)

def parallel_get_node_data_wrapper(args):
    try:
        return parallel_get_node_data(*args)
    except Exception as e:
        print(f"Error processing node with args {args}: {e}")
        return None


def sample_graph_nodes_creation(x_train, y_train, x_test, sample_graphs, processes, 
                                disable=False, verbose=0):
    '''
    Creates and adds node data to sample graphs in parallel.
    '''
    if verbose>0: print('Starting sample graph node creation')
    graph = list(sample_graphs.values())[0]
    all_nodes = list(graph.nodes())
    # Initialize dict to hold information about all nodes, will be of type:
    #   {'sample_id1': {'node1': {'pos_likelihood': -1.58, 'neg_likelihood': -1.86}, 
    #   'node2': {'pos_likelihood': -1.31, 'neg_likelihood': -1.16}}, 
    #   'sample_id2': {'node1': ...}
    sample_nodes_dict = {}
    for ind in x_test.index:
        sample_nodes_dict[ind] = {}
    sample_nodes_weights = {}
    # Generator with parameters for parallel function
    parallel_args = ((n, x_train, y_train, x_test) for n in all_nodes)
    total_number_nodes = len(all_nodes)
    # Create pool of processes for parallelization
    with Pool(processes=processes) as pool:
        # Create iterator for multiprocessing with function and arguments for each call
        results_iterator = pool.imap_unordered(parallel_get_node_data_wrapper, parallel_args)
        # Iterate over iterator
        for result in tqdm(results_iterator, total=total_number_nodes, disable=disable):
            node, nodes_dict, weight = result
            sample_nodes_weights[node] = weight
            for k,v in nodes_dict.items():
                    sample_nodes_dict[k][node] = v
    sample_graphs = gc.add_node_data_to_samples(sample_graphs, sample_nodes_dict, sample_nodes_weights)
    if verbose>0: print('Ending sample graph node creation')
    return sample_graphs


###########################################################################################
########### CLASSIFICATION FUNCTIONS ######################################################
###########################################################################################

def classification_results(sample_graphs, y_test, classification_types: list[str], 
                           weights: list[str], weight_scalings: list[str], disable=False, 
                           verbose=0):
    '''
    Performs parallel classification on sample graphs with various parameter combinations and returns the results.
    '''
    if verbose>0: print('Starting classification')
    # Get possible parameterizations
    params = list(itertools.product(classification_types, weights, weight_scalings))
    total_number_params = len(params)
    # Generator with data and parameterizations 
    parallel_args = ( (sample_graphs, y_test, p[0], p[1], p[2]) for p in params )
    # List to save results
    results_list = []
    for args in tqdm(parallel_args, total=total_number_params, disable=disable):
        result = parallel_graph_classification_wrapper(args)
        results_list.append(result)
    if verbose>0: print('Ending classification')
    return results_list


def parallel_graph_classification(sample_graphs, y_test, classification_type, weighted, weight_scaling):
    '''
    Performs graph classification on sample graphs, computes classification metrics, and returns the results.
    '''
    test_sample_graphs = sample_graphs.copy()
    # Scale Weights
    if weight_scaling is not None and weight_scaling!='None':
        to_minmax = True
        normalize = False
        mymin, mymax = map(float, weight_scaling.split('-'))
    else:
        weight_scaling = 'None'
        to_minmax = False
        normalize = True
        mymin, mymax = None, None
    temp_sample_graphs = gc.transform_weights(test_sample_graphs, where='edge', inverse=False, normalize=normalize, 
                                            minmax=to_minmax, mymin=mymin, mymax=mymax, verbose=0)
    temp_sample_graphs = gc.transform_weights(temp_sample_graphs, where='node', inverse=False, normalize=normalize, 
                                            minmax=to_minmax, mymin=mymin, mymax=mymax, verbose=0)
    true_labels = []
    predictions = []
    probabilities = []
    weight_attr = 'weight' if weighted else None
    for sample_id, g in temp_sample_graphs.items():
        pred, proba = gc.classify_sample_graph(g, classification=classification_type, weight_attr=weight_attr)
        pred = int(pred)
        predictions.append(pred)
        true_labels.append(y_test[sample_id])
        probabilities.append(proba)

    probabilities = np.array(probabilities, dtype='float32')

    if len(np.unique(true_labels)) > 2:
        acc = accuracy_score(true_labels, predictions)
        f1_macro = f1_score(true_labels, predictions, zero_division=0, average='macro')
        rec_macro = recall_score(true_labels, predictions, zero_division=0, average='macro')
        prec_macro = precision_score(true_labels, predictions, zero_division=0, average='macro')

        f1_micro = f1_score(true_labels, predictions, zero_division=0, average='micro')
        rec_micro = recall_score(true_labels, predictions, zero_division=0, average='micro')
        prec_micro = precision_score(true_labels, predictions, zero_division=0, average='micro')

        f1_weighted = f1_score(true_labels, predictions, zero_division=0, average='weighted')
        rec_weighted = recall_score(true_labels, predictions, zero_division=0, average='weighted')
        prec_weighted = precision_score(true_labels, predictions, zero_division=0, average='weighted')

        auc_ovo = roc_auc_score(true_labels, probabilities, multi_class='ovo')
        auc_ovr = roc_auc_score(true_labels, probabilities, multi_class='ovr')
        
        res = {'classification_type': classification_type, 'weighted': weighted, 'minmax': weight_scaling, 
            'accuracy': acc, 'f1_score': f1_macro, 'recall': rec_macro, 'precision': prec_macro,
             'f1_score_micro': f1_micro, 'recall_micro': rec_micro, 'precision_micro': prec_micro,
              'f1_score_weighted': f1_weighted, 'recall_weighted': rec_weighted, 'precision_weighted': prec_weighted,
              'auc_ovo': auc_ovo, 'auc': auc_ovr}

    else:
        acc = accuracy_score(true_labels, predictions)
        f1 = f1_score(true_labels, predictions, zero_division=0, average='binary')
        rec = recall_score(true_labels, predictions, zero_division=0, average='binary')
        prec = precision_score(true_labels, predictions, zero_division=0, average='binary')
        auc = roc_auc_score(true_labels, probabilities[:,1])
        
        res = {'classification_type': classification_type, 'weighted': weighted, 'minmax': weight_scaling, 
            'accuracy': acc, 'f1_score': f1, 'recall': rec, 'precision': prec, 'auc': auc}
        
    return res

def parallel_graph_classification_wrapper(args):
    return parallel_graph_classification(*args)


def get_best_parameterization(df, decision_cols: list[str], verbose=0):
    """
    Selects the best parameterization in the dataframe based on the maximum values of specified decision columns.
    """
    group_keys = ['project', 'layer', 'target', 'pvalue', 'classification_type', 'weighted', 'minmax','dge_filtering']
    # Get mean results across al folds
    df = df.fillna({n: 0.0 if n not in group_keys else 'N/A' for n in df.columns})
    df = df.groupby(by=group_keys).mean()
    df = df.reset_index(group_keys)
    # Choose best parameterization based on validation results
    for c in decision_cols:
        df = df[df[c] == df[c].max()] 
        if df.shape[0] == 1:
            break
    df = df.iloc[0,:]
    if verbose>0: print(f'\nBest parameterization:\n{df}')
    return df

###########################################################################################
########### ML MODELS FUNCTIONS ##########################################################
###########################################################################################

def train_and_test_classifier(classifier, x_train, y_train, x_test, y_test):
        """
        Trains and tests a ML classifier with holdout
        Return dict with metrics
        """
        metrics = {
        'accuracy': None,
        'f1_score': None,
        'recall': None,
        'precision': None,
        'f1_score_micro': None,
        'recall_micro': None,
        'precision_micro': None,
        'f1_score_weighted': None,
        'recall_weighted': None,
        'precision_weighted': None,
        'auc_ovo': None,
        'auc': None,
    }

        classifier.fit(x_train, y_train)
        predictions = classifier.predict(x_test)
        probabilities = classifier.predict_proba(x_test)

        if len(np.unique(y_test)) > 2:  # Multiclass
            metrics['accuracy'] = accuracy_score(y_test, predictions)
            metrics['f1_score'] = f1_score(y_test, predictions, zero_division=0, average='macro')
            metrics['recall'] = recall_score(y_test, predictions, zero_division=0, average='macro')
            metrics['precision'] = precision_score(y_test, predictions, zero_division=0, average='macro')

            metrics['f1_score_micro'] = f1_score(y_test, predictions, zero_division=0, average='micro')
            metrics['recall_micro'] = recall_score(y_test, predictions, zero_division=0, average='micro')
            metrics['precision_micro'] = precision_score(y_test, predictions, zero_division=0, average='micro')

            metrics['f1_score_weighted'] = f1_score(y_test, predictions, zero_division=0, average='weighted')
            metrics['recall_weighted'] = recall_score(y_test, predictions, zero_division=0, average='weighted')
            metrics['precision_weighted'] = precision_score(y_test, predictions, zero_division=0, average='weighted')

            metrics['auc_ovo'] = roc_auc_score(y_test, probabilities, multi_class='ovo')
            metrics['auc'] = roc_auc_score(y_test, probabilities, multi_class='ovr')
        else:  # Binary classification
            metrics['accuracy'] = accuracy_score(y_test, predictions)
            metrics['f1_score'] = f1_score(y_test, predictions, zero_division=0, average='binary')
            metrics['recall'] = recall_score(y_test, predictions, zero_division=0, average='binary')
            metrics['precision'] = precision_score(y_test, predictions, zero_division=0, average='binary')
            metrics['auc'] = roc_auc_score(y_test, probabilities[:, 1])

        return metrics
   

def train_and_test_classifier_cv(classifier, x, y, fold_output_file, specs, k=5):
    """
    Trains and tests a sklearn ML classifier with 5-fold cross-validation.
    Computes metrics including accuracy, F1-score, recall, precision, and AUC.
    Returns averaged metrics.
    """
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    metrics = {
        'accuracy': [],
        'f1_score': [],
        'recall': [],
        'precision': [],
        'f1_score_micro': [],
        'recall_micro': [],
        'precision_micro': [],
        'f1_score_weighted': [],
        'recall_weighted': [],
        'precision_weighted': [],
        'auc_ovo': [],
        'auc': [],
    }
    
    fold = 0
    for train_index, test_index in skf.split(x, y):
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]

        classifier.fit(x_train, y_train)
        predictions = classifier.predict(x_test)
        probabilities = classifier.predict_proba(x_test)

        if len(np.unique(y)) > 2:  # Multiclass
            metrics['accuracy'].append(accuracy_score(y_test, predictions))
            metrics['f1_score'].append(f1_score(y_test, predictions, zero_division=0, average='macro'))
            metrics['recall'].append(recall_score(y_test, predictions, zero_division=0, average='macro'))
            metrics['precision'].append(precision_score(y_test, predictions, zero_division=0, average='macro'))

            metrics['f1_score_micro'].append(f1_score(y_test, predictions, zero_division=0, average='micro'))
            metrics['recall_micro'].append(recall_score(y_test, predictions, zero_division=0, average='micro'))
            metrics['precision_micro'].append(precision_score(y_test, predictions, zero_division=0, average='micro'))

            metrics['f1_score_weighted'].append(f1_score(y_test, predictions, zero_division=0, average='weighted'))
            metrics['recall_weighted'].append(recall_score(y_test, predictions, zero_division=0, average='weighted'))
            metrics['precision_weighted'].append(precision_score(y_test, predictions, zero_division=0, average='weighted'))

            metrics['auc_ovo'].append(roc_auc_score(y_test, probabilities, multi_class='ovo'))
            metrics['auc'].append(roc_auc_score(y_test, probabilities, multi_class='ovr'))
        else:  # Binary classification
            metrics['accuracy'].append(accuracy_score(y_test, predictions))
            metrics['f1_score'].append(f1_score(y_test, predictions, zero_division=0, average='binary'))
            metrics['recall'].append(recall_score(y_test, predictions, zero_division=0, average='binary'))
            metrics['precision'].append(precision_score(y_test, predictions, zero_division=0, average='binary'))
            metrics['auc'].append(roc_auc_score(y_test, probabilities[:, 1]))
        
        current_fold_results = {k : metrics[k][-1] if metrics[k] else 0 for k in metrics.keys()}
        current_fold_results['fold'] = fold+1
        current_fold_results.update(specs)
        save_results(current_fold_results, fold_output_file, ['project', 'layer', 'target', 'classifier', 'fold'])
        
        fold += 1

    # Average the metrics across folds
    averaged_results = {key: np.mean(value) for key, value in metrics.items() if value}
    deviation_results = {key+"_std": np.std(value) for key, value in metrics.items() if value}
    return averaged_results, deviation_results
