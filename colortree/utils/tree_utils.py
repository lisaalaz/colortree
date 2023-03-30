import copy
import math
import numpy as np

import data_utils
import eval_utils


def decision_tree(data, depth):
    '''Takes a dataset and a depth value as inputs (depth value initialised to 0 to
    start building the tree from the root) and returns a binary tree stored as a
    dictionary of dicionaries and its total height. It is recursively applied to
    each new subset which thus becomes the root of a new subtree of the tree.'''
    if np.all(data[1:, -1] == data[0, -1]):
        return (data[0, -1], depth)
    else:
        attribute, value, l_dataset, r_dataset = find_split(data)
        majority_cl = majority_class(data)
        l_branch, l_depth = decision_tree(l_dataset, depth + 1)
        r_branch, r_depth = decision_tree(r_dataset, depth + 1)
        node = {'attribute': attribute, 'checked_prune': False, 'value': value, 'depth': depth,
                'majority_class': majority_cl, 'left': l_branch, 'right': r_branch}
        return (node, max(l_depth, r_depth))


def majority_class(data):
    '''Returns the majority class label in a dataset.'''
    classes = list(data[:, -1])
    class_count = dict.fromkeys(set(classes), 0)
    for i in class_count.keys():
        class_count[i] = classes.count(i)
    max_class = 0
    max_value = 0
    for c, count in class_count.items():
        if count > max_value:
            max_value = count
            max_class = c
    return max_class


def find_split(data):
    '''Finds the best possible split of a multi-dimensional dataset, given as input. 
       It tests multiple split points within and across dimensions, and chooses the
       split that gives the greatest informations gain. Returns the dimension on which 
       the dataset was split, the split point value and the two resulting sub-datasets.
    '''
    best_attribute = 0
    best_gain = 0
    best_splitvalue = 0
    best_l_dataset = 0
    best_r_dataset = 0
    for attribute in range(data.shape[1] - 1):
        sorted_data = data[np.argsort(data[:, attribute])]
        for i, j in zip(np.unique(sorted_data[:, attribute]),
                        np.unique(sorted_data[1:, attribute])):
            splitvalue = (i + j)/2
            l_dataset = split(
                sorted_data, sorted_data[:, attribute] < splitvalue)[0]
            r_dataset = split(
                sorted_data, sorted_data[:, attribute] < splitvalue)[1]
            gain = compute_gain(data, l_dataset, r_dataset)
            if gain > best_gain:
                best_gain = gain
                best_splitvalue = splitvalue
                best_l_dataset = l_dataset
                best_r_dataset = r_dataset
                best_attribute = attribute
    return (best_attribute, best_splitvalue, best_l_dataset, best_r_dataset)


def split(data, value):
    '''Takes an array and a value as input, and returns two arrays split on that value.'''
    return [data[value], data[~value]]


def compute_gain(data, l_data, r_data):
    '''Computes and returns the information gain of a certain dataset split given
       the parent dataset and the two children subsets resulting from the split.
    '''
    total_size = data.shape[0]
    l_size = l_data.shape[0]
    r_size = r_data.shape[0]
    gain = compute_entropy(data) - (((l_size / total_size) * compute_entropy(l_data))
                                    + ((r_size / total_size) * compute_entropy(r_data)))
    return gain


def compute_entropy(data):
    '''Takes an array dataset as input, and outputs its information entropy.'''
    samples = data.shape[0]
    _, counts = np.unique(data[:, -1], return_counts=True)
    entropy = 0
    for i in range(len(counts)):
        if counts[i] > 0:
            entropy += np.negative((counts[i] / samples)
                                   * math.log2(counts[i] / samples))
    return entropy


def replace_node(tree_node, validation_set, child, n_classes):
    '''Takes as input a tree_node eligible for the pruning of its left and right children.
    Checks the average accuracy across all classes of the pruned node vs the unpruned 
    node on the data subset. If accuracy is improved we replace the node with the majority class.
    We reset the 'check_pruned' flag in the original tree to False if we make changes, as the 
    parent node could be eligible for further pruning.'''
    change = 0
    if validation_set.shape[0] == 0:
        return tree_node, change
    unpruned_pred = data_utils.predict_dataset_classes(
        tree_node[str(child)], validation_set)
    unpruned_conf_mat = eval_utils.norm_confusion_matrix(
        eval_utils.confusion_matrix(validation_set, unpruned_pred, n_classes))
    accuracy_unpruned = eval_utils.overall_accuracy(unpruned_conf_mat, n_classes)

    pruned_node = tree_node[str(child)]['majority_class']
    pruned_pred = data_utils.predict_dataset_classes(
        pruned_node, validation_set)
    pruned_conf_mat = eval_utils.norm_confusion_matrix(
        eval_utils.confusion_matrix(validation_set, pruned_pred, n_classes))
    accuracy_pruned = eval_utils.overall_accuracy(pruned_conf_mat, n_classes)

    if accuracy_pruned >= accuracy_unpruned:
        change = 1
        tree_node[str(child)] = tree_node[str(child)]['majority_class']
        tree_node['checked_prune'] = False
        return tree_node, change
    else:
        change = 0
        tree_node['checked_prune'] = True
        return tree_node, change


def prune(tree_node, validation_data, n_classes):
    '''Implements the pruning of pre-terminal nodes.'''
    if type(tree_node) != dict or validation_data.shape[0] == 0 or tree_node['checked_prune'] == True:
        return
    else:
        validation_data_left = validation_data[validation_data[:, int(
            tree_node['attribute'])] <= tree_node['value']]
        validation_data_right = validation_data[validation_data[:, int(
            tree_node['attribute'])] > tree_node['value']]
        if type(tree_node['left']) == dict and type(tree_node['right']) != dict:
            if type(tree_node['left']['left']) != dict and type(tree_node['left']['right']) != dict:
                tree_node['checked_prune'] = True
                tree_node, _ = replace_node(
                    tree_node, validation_data_left, "left", n_classes)
            else:
                prune(tree_node['left'], validation_data_left, n_classes)
        elif type(tree_node['right']) == dict and type(tree_node['left']) != dict:
            if type(tree_node['right']['left']) != dict and type(tree_node['right']['right']) != dict:
                tree_node['checked_prune'] = True
                tree_node, _ = replace_node(
                    tree_node, validation_data_right, "right", n_classes)
            else:
                prune(tree_node['right'], validation_data_right, n_classes)
        else:
            prune(tree_node['left'], validation_data_left, n_classes)
            prune(tree_node['right'], validation_data_right, n_classes)


def pruning_iteration(original_tree, validation_data, n_classes):
    '''Makes a deep copy of the original tree and iteratively prunes checking whether accuracy improves.'''
    tree_to_prune = copy.deepcopy(original_tree)
    list_of_accuracies = []

    pruned_pred = data_utils.predict_dataset_classes(
        original_tree, validation_data)
    pruned_conf_mat = eval_utils.norm_confusion_matrix(
        eval_utils.confusion_matrix(validation_data, pruned_pred, n_classes))
    accuracy_unpruned = eval_utils.overall_accuracy(pruned_conf_mat, n_classes)

    list_of_accuracies.append(accuracy_unpruned)
    finished = False
    iteration = 0
    while not finished:
        iteration += 1
        prune(tree_to_prune, validation_data, n_classes)
        pruned_pred = data_utils.predict_dataset_classes(
            tree_to_prune, validation_data)
        pruned_conf_mat = eval_utils.norm_confusion_matrix(
            eval_utils.confusion_matrix(validation_data, pruned_pred, n_classes))
        accuracy_pruned = eval_utils.overall_accuracy(pruned_conf_mat, n_classes)
        list_of_accuracies.append(accuracy_pruned)

        if list_of_accuracies[iteration] <= list_of_accuracies[iteration-1]:
            finished = True

    return tree_to_prune, original_tree, iteration
