import utils.data_utils as data_utils
import utils.eval_utils as eval_utils
import utils.plot_utils as plot_utils
import utils.tree_utils as tree_utils


def make_tree(train_data):
    '''Outputs a binary decision tree and its height.

    Args:
        train_data: A txt file containing the attributes and the true class in each row.
          Assumes the true class is the last value in each row, and that classes are 
          expressed as integers.

    Returns:
        A tuple containing a dictionary of dictionaries representing the tree, and 
        an int representing the depth of the tree.
        
        Also plots the tree in xy-space.
    '''
    tree, height = tree_utils.decision_tree(train_data, 0)
    plot_utils.draw_tree((tree, height))
    return tree, height


def make_pruned_tree(train_data, valid_data, n_classes):
    '''Outputs a binary decision tree pruned for optimal accuracy, and its height.

    Args:
        train_data: A txt file containing the attributes and the true class in each row.
          Assumes the true class is the last value in each row, and that classes are 
          expressed as integers.
        valid_data: A txt file with same structure as train_data, containing
          the dataset to use for validation.
        
    Returns:
        A tuple containing a dictionary of dictionaries representing the tree, and 
        an int representing the depth of the tree.
        
        Also plots the tree in xy-space.
    '''
    tree, _ = tree_utils.decision_tree(train_data, 0)
    pruned_tree, _, _ = tree_utils.pruning_iteration(tree, valid_data, n_classes)
    height = eval_utils.max_depth(pruned_tree)
    plot_utils.draw_tree((tree, height))
    return pruned_tree, height


def predict_sample(tree, sample):
    '''Does inference on one given sample.

    Args:
        tree: A dictionary of dictionaries representing a decision tree.
        sample: An array representing a single data row.
        
    Returns:
        The predicted integer class.
    '''
    return data_utils.predict_class_of_sample(tree, sample)


def predict_test_set(tree, test_set):
    '''Does inference on one given sample.

    Args:
        tree: A dictionary of dictionaries representing a decision tree.
        test_set: A txt file where each row contains the attributes of a sample.
        
    Returns:
        A list of integers representing the predicted classes of the test set.
    '''
    return data_utils.predict_dataset_classes(tree, test_set)


def eval_report(tree, test_set, n_classes, norm=False):
    '''Prints an evaluation report for a given decision tree.

    Args:
        tree: A dictionary of dictionaries representing a decision tree.
        test_set: A txt file containing the he dataset to be used for testing,
          with the attributes and the true class in each row. Assumes the true
          class is the last value in each row, and that classes are expressed
          as integers.
        n_classes: The number of unique classes in the dataset. Assumes that classes
          are indicated by integers starting from 1.
        norm: Whether the confusion matrix should be normalized. Default is False.        
    '''
    pred_list = data_utils.predict_dataset_classes(tree, test_set)
    conf_matrix = eval_utils.confusion_matrix(test_set, pred_list, n_classes)
    if norm:
        conf_matrix = eval_utils.norm_confusion_matrix(conf_matrix)
    acc = eval_utils.overall_accuracy(conf_matrix, n_classes)
    precision_scores = {}
    recall_scores = {}
    f1_scores = {}

    for c in range(1, n_classes + 1):
        precision_scores[c] = eval_utils.precision(conf_matrix, c, n_classes)
        recall_scores[c] = eval_utils.recall(conf_matrix, c, n_classes)
        f1_scores[c] = eval_utils.f_score(conf_matrix, c, n_classes, beta=1)
        print(f"Metrics for class {c}\nprecision: {precision_scores[c]}  recall: {recall_scores[c]}  f1: {f1_scores[c]}")
    
    print(f"Accuracy (overall): {acc}")
    print(f"Confusion matrix:\n{conf_matrix}")

    








