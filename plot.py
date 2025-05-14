import matplotlib.pyplot as plt

# Define global variables for tree plotting
decision_node = dict(boxstyle="sawtooth", fc="0.8")
leaf_node = dict(boxstyle="round4", fc="0.8")
arrow_args = dict(arrowstyle="<-")

def plot_node(node_text, center_pt, parent_pt, node_type):
    """
    Plot a single node with an arrow pointing to it.
    """
    plt.annotate(node_text, xy=parent_pt, xycoords='axes fraction',
                 xytext=center_pt, textcoords='axes fraction',
                 va="center", ha="center", bbox=node_type, arrowprops=arrow_args)

def get_num_leafs(tree):
    """
    Get the number of leaf nodes in the tree.
    """
    num_leafs = 0
    root = list(tree.keys())[0]
    child_nodes = tree[root]
    for key in child_nodes.keys():
        if isinstance(child_nodes[key], dict):
            num_leafs += get_num_leafs(child_nodes[key])
        else:
            num_leafs += 1
    return num_leafs

def get_tree_depth(tree):
    """
    Get the depth of the tree.
    """
    max_depth = 0
    root = list(tree.keys())[0]
    child_nodes = tree[root]
    for key in child_nodes.keys():
        if isinstance(child_nodes[key], dict):
            this_depth = 1 + get_tree_depth(child_nodes[key])
        else:
            this_depth = 1
        max_depth = max(max_depth, this_depth)
    return max_depth

def plot_mid_text(center_pt, parent_pt, text):
    """
    Plot text in the middle of a branch.
    """
    x_mid = (parent_pt[0] + center_pt[0]) / 2.0
    y_mid = (parent_pt[1] + center_pt[1]) / 2.0
    plt.text(x_mid, y_mid, text, va="center", ha="center", rotation=30)

def plot_tree(tree, parent_pt, node_text):
    """
    Recursively plot the tree.
    """
    num_leafs = get_num_leafs(tree)
    root = list(tree.keys())[0]
    center_pt = (plot_tree.x_off + (1.0 + float(num_leafs)) / 2.0 / plot_tree.total_w, plot_tree.y_off)
    plot_mid_text(center_pt, parent_pt, node_text)
    plot_node(root, center_pt, parent_pt, decision_node)
    child_nodes = tree[root]
    plot_tree.y_off = plot_tree.y_off - 1.0 / plot_tree.total_d
    for key in child_nodes.keys():
        if isinstance(child_nodes[key], dict):
            plot_tree(child_nodes[key], center_pt, key)
        else:
            plot_tree.x_off = plot_tree.x_off + 1.0 / plot_tree.total_w
            plot_node(child_nodes[key], (plot_tree.x_off, plot_tree.y_off), center_pt, leaf_node)
            plot_mid_text((plot_tree.x_off, plot_tree.y_off), center_pt, key)
    plot_tree.y_off = plot_tree.y_off + 1.0 / plot_tree.total_d

def create_plot(tree):
    """
    Create the plot for the decision tree.
    """
    fig = plt.figure(1, facecolor='white')
    fig.clf()
    axprops = dict(xticks=[], yticks=[])
    create_plot.ax1 = plt.subplot(111, frameon=False, **axprops)
    plot_tree.total_w = float(get_num_leafs(tree))
    plot_tree.total_d = float(get_tree_depth(tree))
    plot_tree.x_off = -0.5 / plot_tree.total_w
    plot_tree.y_off = 1.0
    plot_tree(tree, (0.5, 1.0), '')
    plt.show()