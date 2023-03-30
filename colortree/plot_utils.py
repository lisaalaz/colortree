import matplotlib.pyplot as plt

from matplotlib.collections import LineCollection
from matplotlib import colors as mcolors


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
