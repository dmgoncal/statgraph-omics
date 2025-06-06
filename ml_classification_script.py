import argparse
import sys
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from functions import check_ml_result_exists, get_data, save_results, train_and_test_classifier
from constants import DATA_DIR, ML_RESULTS_FILE, RS


TEST_SIZE = 0.3
OUTPUT_FILE = ML_RESULTS_FILE

### Parse arguments ########################################################################################
parser = argparse.ArgumentParser(description="Run graph classification in choosen TCGA data and store results")
parser.add_argument("project", help="TCGA project to use", type=str.upper)
parser.add_argument("layer", help="Layer of omics to use", type=str.lower, choices=['mirna', 'mrna', 'protein'])
parser.add_argument("target", help="Variable to set as target", type=str.lower, choices=['vital_status', 'primary_diagnosis'])

parser.add_argument("-v", "--verbosity", action="count", default=0, help="Increase output verbosity")
parser.add_argument("-r", "--reduced", type=int, help="Reduce the number of features to the specified number") 
parser.add_argument("-s", "--skip", action="store_true", help="Skip already calculated results") 
parser.add_argument("-y", "--yes", action="store_true", help="Recalculate results") 
parser.add_argument("-d", "--dge", type=int, help="p-value (float) or number of genes (int) to use in DGE filtering")

args = parser.parse_args()

# Atribute args to variables
project = args.project
layer = args.layer
target = args.target
# Flags
reduced = args.reduced
verbose = args.verbosity
skip = args.skip
yes = args.yes
dge = args.dge

# Classifiers to use
classifiers = {'rf': RandomForestClassifier(random_state=RS), 
               'lr': LogisticRegression(random_state=RS, class_weight='balanced'), 
               'nb': GaussianNB()}

# Columns to check for results already calculated
results_to_check = {'project': project, 'layer': layer, 'target': target}
# Subset of columns to use when droping duplicates befores saving
drop_duplicates_subset = ['project', 'layer', 'target', 'classifier']


def main():
    command = ' '.join(sys.argv)
    if verbose>0: 
        print(f'\n#####\nRunning: {command}\n#####\n')
    # check if results are already calculated and exit or not according to flags
    classifiers_to_use = check_ml_result_exists(classifiers, results_to_check, OUTPUT_FILE, 
                                                skip, yes)
    # get and process data
    data, features = get_data(DATA_DIR, project, layer, target, dge, reduced, verbose)
    # divide in train and test
    x_train, x_test, y_train, y_test = train_test_split(data[features], data[target], test_size=TEST_SIZE, 
                                                    random_state=RS, stratify=data[target])
    if verbose>0: 
        print(f'Train: {x_train.shape[0]}\nTest: {x_test.shape[0]}')
    del data
    # Train and test each classifier
    for key, classif in classifiers_to_use.items():
        result = train_and_test_classifier(classif, x_train, y_train, x_test, y_test)
        result.update({'project': project, 'layer': layer, 'target': target, 'classifier': key, 'dge_filtering': str(dge)})
        save_results(result, OUTPUT_FILE, drop_duplicates_subset)


if __name__ == "__main__":
    main()
    