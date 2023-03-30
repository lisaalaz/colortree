import numpy as np

import data_utils as data_utils
import tree_utils as tree_utils


def count_classes(i, j, data, pred_class_list):
    '''Counts the pairs i, j of actual and predicted classes (respectively) 
    in a data set. Returns the count as an integer.'''
    count = 0
    for k in range(data.shape[0]):
        if data[k][-1] == i and pred_class_list[k] == j:
            count = count + 1
    return count


def confusion_matrix(data, pred_list, n_classes):
    '''Computes the confusion matrix.'''
    conf_matrix = np.zeros((n_classes, n_classes))
    for i in range(n_classes):
        for j in range(n_classes):
            conf_matrix[i][j] = count_classes(i+1, j+1, data, pred_list)
    return conf_matrix


def norm_confusion_matrix(conf_matrix):
    '''Computes the normalised confusion matrix.'''
    norm_matrix = np.zeros_like(conf_matrix, dtype=float)
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            if sum(conf_matrix[i]) != 0:
                norm_matrix[i][j] = conf_matrix[i][j] / \
                    sum(conf_matrix[i])
    return norm_matrix


def accuracy(conf_matrix, c, n_classes):
    '''Computes the accuracy for a selected class c and returns it as a float.'''
    tp = conf_matrix[int(c) - 1][int(c) - 1]
    tn = 0
    for i in [number for number in range(n_classes) if number != int(c) - 1]:
        for j in [number for number in range(n_classes) if number != int(c) - 1]:
            tn = tn + conf_matrix[i][j]
    total_samples = np.sum(conf_matrix)
    return ((tp + tn) / total_samples)


def overall_accuracy(conf_matrix, n_classes):
    '''Computes the overall accuracy over all classes and returns it as a float.
    A note of caution: it works well for balanced data sets but can be misleading 
    for imbalanced ones.'''
    accuracy_list = []
    for i in range(1, n_classes + 1):
        accuracy_list.append(accuracy(conf_matrix, i, n_classes))
    return sum(accuracy_list) / len(accuracy_list)


def precision(conf_matrix, c, n_classes):
    '''Computes the precision for a selected class c and returns it as a float.'''
    tp = conf_matrix[int(c) - 1][int(c) - 1]
    fp = 0
    for i in [number for number in range(n_classes) if number != int(c) - 1]:
        fp = fp + conf_matrix[int(c) - 1][i]
    return tp / (tp + fp)


def recall(conf_matrix, c, n_classes):
    '''Computes the recall for a selected class c and returns it as a float.'''
    tp = conf_matrix[int(c)-1][int(c)-1]
    fn = 0
    for j in [number for number in range(n_classes) if number != int(c) - 1]:
        fn = fn + conf_matrix[j][int(c) - 1]
    return tp / (tp + fn)


def f_score(conf_matrix, c, n_classes, beta):
    '''Computes the F-score for a selected class and returns it as a float. This is 
    the general version of the F-score formula, to calculate F1 just input weight
    parameter beta = 1.'''
    precision_score = precision(conf_matrix, c, n_classes)
    recall_score = recall(conf_matrix, c, n_classes)
    return (1 + (beta**2)) * (
        precision_score*recall_score) / (((beta**2) * precision_score + recall_score))


def max_depth(tree, depth=0):
    '''Returns the maximum depth of a tree.'''
    if type(tree['left']) != dict and type(tree['right']) != dict:
        depth = depth+1
        return depth
    elif type(tree['left']) == dict and type(tree['right']) != dict:
        depth = tree['left']['depth']
        tree = tree['left']
        return max_depth(tree, depth)
    elif type(tree['right']) == dict and type(tree['left']) != dict:
        depth = tree['right']['depth']
        tree = tree['right']
        return max_depth(tree, depth)
    else:
        depth = depth + 1
        l_depth = max_depth(tree['left'], depth)
        r_depth = max_depth(tree['right'], depth)
        if l_depth > r_depth:
            return l_depth
        else:
            return r_depth


def k_fold_evaluation(list_of_k_fold, n_classes):
    '''Evaluation function for cross-validation.'''
    unpruned_accuracy = np.zeros(10)
    pruned_accuracy = np.zeros(10)
    unpruned_accuracy_val = np.zeros(10)
    pruned_accuracy_val = np.zeros(10)

    unpruned_conf_matrix_norm_total = np.zeros((n_classes, n_classes))
    pruned_conf_matrix_norm_total = np.zeros((n_classes, n_classes))
    unpruned_conf_matrix_norm_total_val = np.zeros((n_classes, n_classes))
    pruned_conf_matrix_norm_total_val = np.zeros((n_classes, n_classes))

    for test_number in range(10):
        test_set, training_set_total = data_utils.cross_validation_sets(
            test_number, list_of_k_fold)
        list_of_val_train = data_utils.k_fold(training_set_total, 9)
        val_data, training_set = data_utils.cross_validation_sets(
            0, list_of_val_train)
        tree_trained = tree_utils.decision_tree(training_set, depth=0)
        prune_tree, original_tree, _ = tree_utils.pruning_iteration(
            tree_trained[0], val_data, n_classes)

        unpruned_pred_class = data_utils.predict_dataset_classes(
            original_tree, test_set)
        pruned_pred_class = data_utils.predict_dataset_classes(
            prune_tree, test_set)
        unpruned_pred_class_val = data_utils.predict_dataset_classes(
            original_tree, val_data)
        pruned_pred_class_val = data_utils.predict_dataset_classes(
            prune_tree, val_data)

        norm_conf_matrix_unpruned = norm_confusion_matrix(
            confusion_matrix(test_set, unpruned_pred_class, n_classes))
        norm_conf_matrix_pruned = norm_confusion_matrix(
            confusion_matrix(test_set, pruned_pred_class, n_classes))
        norm_conf_matrix_unpruned_val = norm_confusion_matrix(
            confusion_matrix(val_data, unpruned_pred_class_val, n_classes))
        norm_conf_matrix_pruned_val = norm_confusion_matrix(
            confusion_matrix(val_data, pruned_pred_class_val, n_classes))

        unpruned_conf_matrix_norm_total += norm_conf_matrix_unpruned
        pruned_conf_matrix_norm_total += norm_conf_matrix_pruned
        unpruned_conf_matrix_norm_total_val += norm_conf_matrix_unpruned_val
        pruned_conf_matrix_norm_total_val += norm_conf_matrix_pruned_val

        unpruned_accuracy[test_number] = overall_accuracy(
            norm_conf_matrix_unpruned)
        pruned_accuracy[test_number] = overall_accuracy(
            norm_conf_matrix_pruned)
        unpruned_accuracy_val[test_number] = overall_accuracy(
            norm_conf_matrix_unpruned_val)
        pruned_accuracy_val[test_number] = overall_accuracy(
            norm_conf_matrix_pruned_val)

    unpruned_final_conf_matrix_norm = unpruned_conf_matrix_norm_total*10
    pruned_final_conf_matrix_norm = pruned_conf_matrix_norm_total*10
    unpruned_final_conf_matrix_norm_val = unpruned_conf_matrix_norm_total_val*10
    pruned_final_conf_matrix_norm_val = pruned_conf_matrix_norm_total_val*10

    total_unpruned_acc = np.average(unpruned_accuracy)*100
    total_pruned_acc = np.average(pruned_accuracy)*100
    total_unpruned_acc_val = np.average(unpruned_accuracy_val)*100
    total_pruned_acc_val = np.average(pruned_accuracy_val)*100

    return (unpruned_final_conf_matrix_norm, 
            pruned_final_conf_matrix_norm, 
            total_unpruned_acc, 
            total_pruned_acc, 
            unpruned_final_conf_matrix_norm_val, 
            pruned_final_conf_matrix_norm_val, 
            total_unpruned_acc_val, 
            total_pruned_acc_val)
