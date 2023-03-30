import copy
import math
import matplotlib.pyplot as plt
import numpy as np

from matplotlib.collections import LineCollection
from matplotlib import colors as mcolors


def make_tree(train_data):
    '''Outputs a binary decision tree and its height.

    Args:
        train_data: A bidimensional array containing the attributes and the true class
          in each row. Assumes the true class is the last value in each row, and that 
          classes are expressed as integers.

    Returns:
        A tuple containing a dictionary of dictionaries representing the tree, and 
        an int representing the depth of the tree.
        
        Also plots the tree in xy-space.
    '''
    tree, height = decision_tree(train_data, 0)
    draw_tree((tree, height))
    return tree, height


def make_pruned_tree(train_data, valid_data, n_classes):
    '''Outputs a binary decision tree pruned for optimal accuracy, and its height.

    Args:
        train_data: A bidimensional array containing the attributes and the true class
          in each row. Assumes the true class is the last value in each row, and that 
          classes are expressed as integers.
        valid_data: A txt file with same structure as train_data, containing
          the dataset to use for validation.
        n_classes: The number of unique classes in the data.
        
    Returns:
        A tuple containing a dictionary of dictionaries representing the tree, and 
        an int representing the depth of the tree.
        
        Also plots the tree in xy-space.
    '''
    tree, _ = decision_tree(train_data, 0)
    pruned_tree, _, _ = pruning_iteration(tree, valid_data, n_classes)
    height = max_depth(pruned_tree)
    draw_tree((tree, height))
    return pruned_tree, height


def predict_sample(tree, sample):
    '''Does inference on one given sample.

    Args:
        tree: A dictionary of dictionaries representing a decision tree.
        sample: An array representing a single data row.
        
    Returns:
        The predicted integer class.
    '''
    return predict_class_of_sample(tree, sample)


def predict_test_set(tree, test_set):
    '''Does inference on one given sample.

    Args:
        tree: A dictionary of dictionaries representing a decision tree.
        test_set: A bidimensional array containing the attributes in each row.
        
    Returns:
        A list of integers representing the predicted classes of the test set.
    '''
    return predict_dataset_classes(tree, test_set)


def eval_report(tree, test_set, n_classes, norm=False):
    '''Prints an evaluation report for a given decision tree.

    Args:
        tree: A dictionary of dictionaries representing a decision tree.
        test_set: A bidimensional array containing the attributes and the true class
          in each row. Assumes the true class is the last value in each row, and that 
          classes are expressed as integers.
        n_classes: The number of unique classes in the dataset. Assumes that classes
          are indicated by integers starting from 1.
        norm: Whether the confusion matrix should be normalized. Default is False.        
    '''
    pred_list = predict_dataset_classes(tree, test_set)
    conf_matrix = confusion_matrix(test_set, pred_list, n_classes)
    if norm:
        conf_matrix = norm_confusion_matrix(conf_matrix)
    acc = overall_accuracy(conf_matrix, n_classes)
    precision_scores = {}
    recall_scores = {}
    f1_scores = {}

    for c in range(1, n_classes + 1):
        precision_scores[c] = precision(conf_matrix, c, n_classes)
        recall_scores[c] = recall(conf_matrix, c, n_classes)
        f1_scores[c] = f_score(conf_matrix, c, n_classes, beta=1)
        print(f"Metrics for class {c}\nprecision: {precision_scores[c]}  recall: {recall_scores[c]}  f1: {f1_scores[c]}")
    
    print(f"Accuracy (overall): {acc}")
    print(f"Confusion matrix:\n{conf_matrix}")

    

################################# data utils #################################

def predict_class_of_sample(my_tree, data_line):
    '''Takes a one-dimensional array (i.e. one sample) and classifies it using the 
    decision tree. It returns an integer (the predicted class).'''
    if type(my_tree) == dict:
        if data_line[my_tree['attribute']] < my_tree['value']:
            if type(my_tree['left']) == dict:
                my_new_tree = my_tree['left']
                return predict_class_of_sample(my_new_tree, data_line)
            else:
                return my_tree['left']
        else:
            if type(my_tree['right']) == dict:
                my_new_tree = my_tree['right']
                return predict_class_of_sample(my_new_tree, data_line)
            else:
                return my_tree['right']
    else:
        return my_tree


def predict_dataset_classes(my_tree, data):
    '''Predicts the classes of the data samples of a full data set. Returns a list 
    of integers corresponding to the predicted class of each sample.'''
    predicted_class_lst = []
    n_samples = data.shape[0]
    for i in range(n_samples):
        predicted_class_lst.append(predict_class_of_sample(my_tree, data[i]))
    return predicted_class_lst


################################# tree utils #################################

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
    the dataset was split, the split point value and the two resulting sub-datasets.'''
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
    the parent dataset and the two children subsets resulting from the split.'''
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
    unpruned_pred = predict_dataset_classes(
        tree_node[str(child)], validation_set)
    unpruned_conf_mat = norm_confusion_matrix(
        confusion_matrix(validation_set, unpruned_pred, n_classes))
    accuracy_unpruned = overall_accuracy(unpruned_conf_mat, n_classes)

    pruned_node = tree_node[str(child)]['majority_class']
    pruned_pred = predict_dataset_classes(
        pruned_node, validation_set)
    pruned_conf_mat = norm_confusion_matrix(
        confusion_matrix(validation_set, pruned_pred, n_classes))
    accuracy_pruned = overall_accuracy(pruned_conf_mat, n_classes)

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

    pruned_pred = predict_dataset_classes(
        original_tree, validation_data)
    pruned_conf_mat = norm_confusion_matrix(
        confusion_matrix(validation_data, pruned_pred, n_classes))
    accuracy_unpruned = overall_accuracy(pruned_conf_mat, n_classes)

    list_of_accuracies.append(accuracy_unpruned)
    finished = False
    iteration = 0
    while not finished:
        iteration += 1
        prune(tree_to_prune, validation_data, n_classes)
        pruned_pred = predict_dataset_classes(
            tree_to_prune, validation_data)
        pruned_conf_mat = norm_confusion_matrix(
            confusion_matrix(validation_data, pruned_pred, n_classes))
        accuracy_pruned = overall_accuracy(pruned_conf_mat, n_classes)
        list_of_accuracies.append(accuracy_pruned)

        if list_of_accuracies[iteration] <= list_of_accuracies[iteration-1]:
            finished = True

    return tree_to_prune, original_tree, iteration


################################# eval utils #################################

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


################################# plot utils #################################

def tree_nodes(tree, width_distance, depth_distance):
    '''Converts a tree stored as a dictionary of dictionaries to a representation 
    of its edges and their location on the x-y plane.'''
    all_nodes = []

    def get_nodes(node, offset, width_distance, depth_distance):
        # Appends each node to the global list of nodes with coordinates.
        # Recursive function: starts a node and gets its left and right children.
        if (type({}) == type(node)):
            left_node = {}
            left_node["startpoint"] = {'data': (node['attribute'], node['value']),
                                       'y': offset, 'x': node['depth'] * depth_distance}
            right_node = {}
            right_node["startpoint"] = {'data': (node['attribute'], node['value']),
                                        'y': offset, 'x': node['depth'] * depth_distance}
            left_node["endpoint"] = {'data': get_nodes(node['left'],
                                     offset - width_distance/2, width_distance/2,
                                     depth_distance), 'y': offset - width_distance/2,
                                     'x': depth_distance * (node['depth'] + 1)}
            right_node["endpoint"] = {'data': get_nodes(node['right'],
                                      offset + width_distance/2, width_distance/2,
                                      depth_distance), 'y': offset + width_distance/2,
                                      'x': depth_distance * (node['depth'] + 1)}
            all_nodes.append(left_node)
            all_nodes.append(right_node)
            return (node['attribute'], node['value'])
        else:
            return node

    # all appended nodes are returned by the outer function
    get_nodes(tree, 0, width_distance, depth_distance)
    return all_nodes


def tree_edges(all_nodes):
    '''Returns a litst of coordinates for the start and end points of the edges of 
    the tree, to be plotted on the plane.'''
    edges = []
    for n in all_nodes:
        startX = n['startpoint']['x']
        startY = n['startpoint']['y']
        endX = n['endpoint']['x']
        endY = n['endpoint']['y']
        # the line is made of two points - start and end
        edges.append([[startX, startY], [endX, endY]])
    return edges


def draw_tree(tree):
    '''Plots the tree on the x-y plane. Takes input in the same form output by the 
    decision_tree() function, that is, a tuple with a tree expressed 
    as nested dictionaries as first element and its total height as second element.'''
    width_distance = 1500
    depth_distance = 1000
    figsizeX = 25
    figsizeY = 35
    starting_node = tree[0]
    levels = tree[1]
    all_nodes = tree_nodes(starting_node, width_distance, depth_distance)
    lines = tree_edges(all_nodes)
    colors = [mcolors.to_rgba(c)
              for c in plt.rcParams['axes.prop_cycle'].by_key()['color']]
    line_segments = LineCollection(
        lines, linewidths=1, colors=colors, linestyle='solid')
    fig, ax = plt.subplots(figsize=(figsizeX, figsizeY))
    ax.set_xlim(-1, (levels + 1) * (depth_distance + 1))
    ax.set_ylim(-1 * width_distance, 1 * width_distance)
    ax.add_collection(line_segments)

    for n in all_nodes:
        if (0 == n['startpoint']['x']):
            startX = n['startpoint']['x']
            startY = n['startpoint']['y']
            xy = (startX, startY)
            ax.annotate('(%s, %s)' % n['startpoint']
                        ['data'], xy=xy, textcoords='data')
        endX = n['endpoint']['x']
        endY = n['endpoint']['y']
        xy = (endX, endY)
        if (type(()) == type(n['endpoint']['data'])):
            ax.annotate('(%s, %s)' % n['endpoint']
                        ['data'], xy=xy, textcoords='data')
        else:
            ax.annotate('(%s)' % n['endpoint']['data'],
                        xy=xy, textcoords='data')

    plt.show()

