import argparse
import itertools
import os, sys
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from functions import classification_results, get_best_parameterization, get_data, parallel_graph_classification_wrapper, prune_sample_graphs, sample_graph_edges_creation, sample_graph_nodes_creation, save_results
from functions import check_result_exists
from constants import DATA_DIR, CV_RESULTS_FILE, CV_FOLD_RESULTS_FILE, RS


N_SPLITS = 5
TEST_SIZE = 0.3
OUTPUT_FILE = CV_RESULTS_FILE
FOLD_OUTPUT_FILE = CV_FOLD_RESULTS_FILE
DISABLE_PROGRESS_BAR = False

### Parse arguments ########################################################################################
parser = argparse.ArgumentParser(description="Run graph classification in choosen TCGA data with CV hyperparameterization and store results")
parser.add_argument("project", help="TCGA project to use", type=str.upper)
parser.add_argument("layer", help="Layer of omics to use", type=str.lower, choices=['mirna', 'mrna', 'protein'])
parser.add_argument("target", help="Variable to set as target", type=str.lower, choices=['vital_status', 'primary_diagnosis'])
parser.add_argument("processes", help="Number (int) or fraction (float) of processors to use")
# Flags
parser.add_argument("-d", "--dge", type=float,  help="(>1) number of genes | (<1) p-value -- to use in DGE filtering")
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
dge = args.dge
reduced = args.reduced
verbose = args.verbosity
skip = args.skip
yes = args.yes
processes = int(processes*os.cpu_count()) if isinstance(processes, float) else processes

# Columns to check for results already calculated
results_to_check = {'project': project, 'layer': layer, 'target': target}
# Columns to save
result_columns = ['project', 'layer', 'target', 'pvalue', 'classification_type', 'weighted', 'minmax', 'accuracy', 'f1_score', 'recall', 'precision', 'auc','auc_ovo', 'f1_score_micro', 'recall_micro', 'precision_micro', 'f1_score_weighted', 'recall_weighted', 'precision_weighted']

# Paremterizations to try
pvalue_values = [0.01,  0.001, 0.0001]
classification_types = ['edge', 'both', 'node']
weights = [True, False]
weight_scalings = [None, '0-1', '0.3-1', '0.5-1', '0.8-1']
# Columns to choose best parameterization (ordered by importance)
decision_cols = ['f1_score', 'recall','precision', 'accuracy']
# Subset of columns to use when droping duplicates befores saving
drop_duplicates_subset = ['project', 'layer', 'target']


def main():
    command = ' '.join(sys.argv)
    if verbose>0: 
        print(f'\n#####\nRunning: {command}\n#####\n')
    # check if results are already calculated and exit or not according to flags
    check_result_exists(results_to_check, OUTPUT_FILE, skip, yes)
    # get and process data
    data, features = get_data(DATA_DIR, project, layer, target, dge, reduced, verbose)
    # divide in train and test
    x_train, x_test, y_train, y_test = train_test_split(data[features], data[target], test_size=TEST_SIZE, 
                                                    random_state=RS, stratify=data[target])
    if verbose>0: 
        print(f'Train: {x_train.shape[0]}\nTest: {x_test.shape[0]}')
    del data

    # Graph Creation
    total_number_edges = len(list(itertools.combinations(features, 2)))
    if verbose>0: 
        print("Total number of variable pairs: ", total_number_edges)
    pvalues = sorted(pvalue_values, reverse=True)
    print(f"Starting with p-value:{pvalues[0]} and pruning after that in the following order: {pvalues[1:]}")

    # Temporary dataframe to save results of different parameterizations
    df_val = pd.DataFrame(columns=result_columns).astype({"weighted":bool})
    # Cross-Validation
    skf = StratifiedKFold(n_splits=N_SPLITS)
    for f, (trn_idx, tst_idx) in enumerate(skf.split(x_train, y_train)):
        if verbose>0: 
            print(f'\n## Fold {f+1} ##')
        # Get train and test data for this fold
        x_fold_train, y_fold_train = x_train.iloc[trn_idx], y_train.iloc[trn_idx]
        x_fold_test, y_fold_test = x_train.iloc[tst_idx], y_train.iloc[tst_idx]
        # Create first graphs with highest p-value
        pvalue = pvalues[0]
        # Create sample graphs with edges
        sample_graphs = sample_graph_edges_creation(x_fold_train, y_fold_train, x_fold_test, pvalue, 
                                                    features, processes, DISABLE_PROGRESS_BAR, verbose)
        # Add node data to sample graphs
        sample_graphs = sample_graph_nodes_creation(x_fold_train, y_fold_train, x_fold_test, 
                                                    sample_graphs, processes, DISABLE_PROGRESS_BAR, verbose)
        # Incresingly prune graphs by reducing p-value threshold
        for pvalue in pvalues:
            if verbose>0: 
                print(f'\n# Pruning with pvalue: {pvalue}')
            # Prune graphs
            sample_graphs = prune_sample_graphs(sample_graphs, pvalue)
            # Check if graphs are empty after prunning
            tgraph = list(sample_graphs.values())[0]
            if len(tgraph.edges) == 0:
                print('Empty graphs, ending pruning')
                break
            # Get classification results with the various parameterizations
            results_list = classification_results(sample_graphs, y_fold_test, 
                                                    classification_types, weights, 
                                                    weight_scalings, DISABLE_PROGRESS_BAR, verbose)
            # Update results with data-specific information and add to DataFrame
            for result in results_list:
                result.update({'project': project, 'layer': layer, 'target': target, 'pvalue': pvalue,
                            'dge_filtering': str(dge)})
                fold_result = result.copy()
                fold_result['fold'] = f+1
                save_results(fold_result, FOLD_OUTPUT_FILE, drop_duplicates_subset+['pvalue','classification_type',
                          'weighted','dge_filtering','minmax','fold'])
                df_val = pd.concat([df_val, pd.DataFrame([result])], ignore_index=True)
    del sample_graphs

    # Test graph classification with best paramaterization
    # Get best parameterization
    best_parameterization = get_best_parameterization(df_val, decision_cols, verbose)
    test_pvalue = best_parameterization['pvalue']
    test_classification_type = best_parameterization['classification_type']
    test_weighted = best_parameterization['weighted']
    test_minmax = best_parameterization['minmax']
    # Create test sample graphs using best parameterization
    test_sample_graphs = sample_graph_edges_creation(x_train, y_train, x_test, test_pvalue, features,
                                                        processes, DISABLE_PROGRESS_BAR, verbose)
    # Add node data to sample graphs
    test_sample_graphs = sample_graph_nodes_creation(x_train, y_train, x_test, test_sample_graphs,
                                                        processes, DISABLE_PROGRESS_BAR, verbose)
    # Classification results
    args = (test_sample_graphs, y_test, test_classification_type, test_weighted, test_minmax)
    final_results = parallel_graph_classification_wrapper(args)
    # add complementary information
    final_results.update({'project': project, 'layer': layer, 'target': target, 'pvalue': test_pvalue,
                'dge_filtering': str(dge)})
    print(f'Final results: {final_results}')
    # Save results in file
    if reduced is None:
        save_results(final_results, OUTPUT_FILE, drop_duplicates_subset)
    else:
        print(final_results)


if __name__ == "__main__":
    main()
