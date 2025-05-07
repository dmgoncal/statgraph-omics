import argparse
import itertools
import os, sys
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from constants import DATA_DIR, RESULTS_FILE, RS
from functions import classification_results, get_data, prune_sample_graphs, sample_graph_edges_creation, sample_graph_nodes_creation, save_results
from functions import check_result_exists


TEST_SIZE = 0.3
OUTPUT_FILE = RESULTS_FILE
DISABLE_PROGRESS_BAR = False

### Parse arguments ########################################################################################
parser = argparse.ArgumentParser(description="Run graph classification in choosen TCGA data and store results")
parser.add_argument("project", help="TCGA project to use", type=str.upper)
parser.add_argument("layer", help="Layer of omics to use", type=str.lower, choices=['mirna', 'mrna', 'protein'])
parser.add_argument("target", help="Variable to set as target", type=str.lower, choices=['vital_status', 'primary_diagnosis'])
parser.add_argument("processes", help="Number (int) or fraction (float) of processors to use")
# Flags
parser.add_argument("-d", "--dge", type=float, help="(>1) number of genes | (<1) p-value -- to use in DGE filtering")
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
# Subset of columns to use when droping duplicates befores saving
drop_duplicates_subset = ['project', 'layer', 'target', 'pvalue','classification_type',
                        'weighted','dge_filtering','minmax']

def main():
    command = ' '.join(sys.argv)
    if verbose>0: 
        print(f'\n#####\nRunning: {command}\n#####\n')
    # check if results are already calculated and exit or not according to flags
    check_result_exists(results_to_check, OUTPUT_FILE, skip, yes)
    # get preprocessed data
    data, features = get_data(DATA_DIR, project, layer, target, dge, reduced, verbose)
    # divide in train and test
    x_train, x_test, y_train, y_test = train_test_split(data[features], data[target], test_size=TEST_SIZE, 
                                                    random_state=RS, stratify=data[target])
    if verbose>0: 
        print(f'Train: {x_train.shape[0]}\nTest: {x_test.shape[0]}')
    # delete data obejct to save memory
    del data

    # get the number of edges that the fully connected graph has
    total_number_edges = len(list(itertools.combinations(features, 2)))
    if verbose>0: 
        print("Total number of variable pairs: ", total_number_edges)
    pvalues = sorted(pvalue_values, reverse=True)
    print(f"Starting with p-value:{pvalues[0]} and pruning after that in the following order: {pvalues[1:]}")
    
    # Create first graphs with highest p-value
    pvalue = pvalues[0]
    # Create sample graphs with edges
    sample_graphs = sample_graph_edges_creation(x_train, y_train, x_test, pvalue, 
                                                features, processes, DISABLE_PROGRESS_BAR, verbose)
    # Add node data to sample graphs
    sample_graphs = sample_graph_nodes_creation(x_train, y_train, x_test, sample_graphs,
                                                processes, DISABLE_PROGRESS_BAR, verbose)
    # Delete large variables with no further use
    del x_train, y_train, x_test
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
        results_list = classification_results(sample_graphs, y_test,
                                            classification_types, weights, 
                                            weight_scalings, DISABLE_PROGRESS_BAR, verbose)

        # Update results with data-specific information and add to DataFrame
        for result in results_list:
            result.update({'project': project, 'layer': layer, 'target': target, 'pvalue': pvalue,
                        'dge_filtering': str(dge)})
            # Save results in file
            if reduced is None:
                save_results(result, OUTPUT_FILE, drop_duplicates_subset)
            else:
                print(result)

if __name__ == "__main__":
    main()
    