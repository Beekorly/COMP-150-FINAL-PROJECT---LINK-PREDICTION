# coding: utf-8
import networkx as nx
import numpy as np
import os

categories = ["tvshow", "government", "company", "politician"]
BASE = os.getcwd() + '/data/musae_facebook_'
step = 5


def get_graph():
    """
    Read edge information from the file musae_facebook_edges.csv file.
    In order to change the location of the file, need to modify BASE.
    :return: an undirected networkx graph
    """
    G = nx.Graph()

    file = open(BASE + 'edges.csv', "r")
    file.readline()  # skip the first line
    line = file.readline()[:-1]
    while line:
        edge = list(map(int, line.split(",")))
        G.add_edge(edge[0], edge[1])
        line = file.readline()[:-1] # next line
    file.close()

    return G


def get_node_info():
    """
    Read node information from the file musae_facebook_target.csv file.
    In order to change the location of the file, need to modify BASE.
    :return: a dictionary linking nodes to a category
    """
    node_info = {}

    file = open(BASE + 'target.csv', "r", encoding='utf-8')
    file.readline()  # skip the first line
    line = file.readline()[:-1]
    while line:
        line = line.split(",")
        if not line[3] in categories:
            node_info[int(line[0])] = line[-1]
        else:
            node_info[int(line[0])] = line[3]
        line = file.readline()[:-1] # next line
    file.close()

    return node_info


def corrected(edge_type):
    """
    Sort the string alphabetically.
    :param edge_type: the edge type string
    :return: the edge type string sorted alphabetically
    """
    return "_".join(sorted(edge_type.split("_")))


def get_edge_type(edge):
    """
    Get edge type string of an edge in G.
    :param edge: a tuple of two integers that are nodes in G
    :return: the edge type string
    """
    return "_".join(sorted([node_categories[edge[0]], node_categories[edge[1]]]))


def get_stats(write = 0):
    """
    Count number of nodes in each category and number of edges in each category pair.
    :param write: a binary 0/1 variable that is used to decide whether the function prints
    :return: N/A
    """
    node_type_counts = {}
    for node in G.nodes:
        if not node_categories[node] in node_type_counts:
            node_type_counts[node_categories[node]] = 1
        else:
            node_type_counts[node_categories[node]] += 1

    # count the number of edges of each node type pair
    edge_type_counts = {}
    for edge in G.edges:
        edge_type = get_edge_type(edge)
        if not edge_type in edge_type_counts:
            edge_type_counts[edge_type] = 1
        else:
            edge_type_counts[edge_type] += 1

    if write:
        for category, count in node_type_counts.items():
            print(category, count)
        for category, count in edge_type_counts.items():
            print(category, count)


def make_copy(H):
    """
    Makes a copy of networkx graph H.
    :param H: the initial networkx graph
    :return: a networkx graph that is a copy of H
    """

    F = nx.Graph()

    for edge in H.edges:
        F.add_edge(edge[0], edge[1])

    return F


def remove_edges(H, edge_type, P):
    """
    Removes edges of a specific type from H based on a percentage.
    :param H: the initial networkx graph
    :param edge_type: the edge type string
    :param P: the probability of removing an edge
    :return: the same networkx graph with some removed edges and the list of removed edges
    """
    removed_edges = []

    edges = list(H.edges)

    for edge in edges:
        if get_edge_type(edge) == edge_type:
            if np.random.random() < P:
                H.remove_edge(edge[0], edge[1])
                removed_edges.append(edge)

    return H, removed_edges


def get_nodes(H, node1_type, node2_type):
    """
    Get two sets of two different node types in a graph.
    :param H: the networkx graph
    :param node1_type: a node type string
    :param node2_type: a node type string
    :return: two sets that contain all edges of each node type
    """
    type1_nodes, type2_nodes = set(), set()

    for node in H.nodes:
        if H.degree[node]:
            if node_categories[node] == node1_type:
                type1_nodes.add(node)
            if node_categories[node] == node2_type:
                type2_nodes.add(node)

    return type1_nodes, type2_nodes


def get_node_pairs(H, type1_nodes, type2_nodes):
    """
    Get a list of all node pairs of two different types, excluding node pairs that are already edges in the graph
    :param H: the networkx graph
    :param type1_nodes: a node type string
    :param type2_nodes: a node type string
    :return: a list of node pair tuples
    """
    node_pairs = []
    for type1_node in type1_nodes:
        for type2_node in type2_nodes:
            if not H.has_edge(type1_node, type2_node):
                node_pairs.append((type1_node, type2_node))

    return node_pairs


def get_generator_values(generator, node_pairs):
    """
    Iterate over a generator and turn it into a list
    :param generator: the iterator
    :param node_pairs: all node pairs, used for tracking progress
    :return: list of tuples of node pairs and values
    """
    removed_edge_weights = []

    percent = step
    counter = 0

    for node1, node2, value in generator:
        counter += 1
        if counter == percent * len(node_pairs) // 100:
            print(str(percent) + "%")
            percent += step
        removed_edge_weights.append((node1, node2, value))

    return removed_edge_weights


def analyze(edges_predicted, write = 0):
    """
    Count the number of correctly predicted edges
    :param edges_predicted: list of edges predicted by the current method
    :param write: a binary 0/1 variable that is used to decide whether the function prints
    :return: number of correctly predicted edges
    """
    correct = 0
    for edge in edges_predicted:
        if G.has_edge(edge[0], edge[1]):
            correct += 1

    if write:
        print("We predicted", len(edges_predicted), "edges.")
        print("Of those,", correct, "were predicted correctly.")
        print("The percentage of edges predicted correctly is", format(correct / len(edges_predicted) * 100, '.5f'))

    return correct


def get_CN_values(H, type1_nodes, type2_nodes):
    """
    Our implementation of couting common neighbors.
    :param H: the networkx graph
    :param type1_nodes: a node type string
    :param type2_nodes: a node type string
    :return: list of tuples of node pairs and values
    """
    removed_edge_weights = []

    percent = step
    counter = 0

    for type1_node in type1_nodes:
        neighbors = [n for n in H.neighbors(type1_node)]
        counter += 1
        if counter == percent * len(type1_nodes) // 100:
            print(str(percent) + "%")
            percent += step
        for type2_node in type2_nodes:
            if not type2_node in neighbors:
                removed_edge_weights.append((type1_node, type2_node, len(list(nx.common_neighbors(H, type1_node, type2_node)))))

    return removed_edge_weights


def run_random_walks(H, type1_nodes, type2_nodes, walks, walk_len):
    """
    Runs random walks from all nodes of one type and counts the number of times nodes of another type are reached.
    :param H: the networkx graph
    :param type1_nodes: the start nodes for random walks
    :param type2_nodes: the target nodes
    :param walks: the number of walks to be done from each node of the first type
    :param walk_len: the number of steps in each walk
    :return: a list of dictionaries where each dictionary represents the reach count for nodes of the second type
    """
    node_reach_counters = []

    for type1_node in type1_nodes:
        node_reaches = {}
        walk_count = 0
        while walk_count < walks:
            current = type1_node
            for step in range(walk_len):
                neighbors = list(H.neighbors(current))
                if len(neighbors):
                    if len(neighbors) == 1:
                        next = neighbors[0]
                    else:
                        next = neighbors[np.random.randint(0, len(neighbors) - 1)]
                else:
                    break

                current = next

                if current in type2_nodes:
                    if not H.has_edge(type1_node, current):
                        if not current in node_reaches:
                            node_reaches[current] = 1
                        else:
                            node_reaches[current] += 1

            walk_count += 1
        node_reach_counters.append(node_reaches)

    return node_reach_counters


def run_walks(H, bi_directed, type1_nodes, type2_nodes, walks, walk_len, removed_edges):
    """
    Runs random walks in one direction, or both directions, and creates the list of tuples of node pairs and values.
    :param H: the networkx graph
    :param bi_directed: a binary T/F value that specifies whether we run random walks from only one node type
    :param type1_nodes: the start nodes for random walks
    :param type2_nodes: the target nodes
    :param walks: the number of walks to be done from each node of the first type
    :param walk_len: the number of steps in each walk
    :param removed_edges: the list of removed edges
    :return: list of tuples of node pairs and values
    """
    if bi_directed:
        node1_reach_counters = run_random_walks(H, list(type1_nodes), type2_nodes, walks, walk_len)
        node2_reach_counters = run_random_walks(H, list(type2_nodes), type1_nodes, walks, walk_len)
        type1_nodes = list(type1_nodes)
        type2_nodes = list(type2_nodes)
    else:
        type1_nodes = list(type1_nodes)
        node_reach_counters = run_random_walks(H, type1_nodes, type2_nodes, walks, walk_len)

    removed_edge_weights = []
    if bi_directed:
        for i in range(len(node1_reach_counters)):
            for type2_node, value in node1_reach_counters[i].items():
                other_value = 0
                j = type2_nodes.index(type2_node)
                if type1_nodes[i] in node2_reach_counters[j]:
                    other_value = node2_reach_counters[j][type1_nodes[i]]
                removed_edge_weights.append((type1_nodes[i], type2_node, value + other_value))
    else:
        for i in range(len(node_reach_counters)):
            for type2_node, value in node_reach_counters[i].items():
                removed_edge_weights.append((type1_nodes[i], type2_node, value))

    removed_edge_weights = sorted(removed_edge_weights, key=lambda x: x[2], reverse=True)
    return removed_edge_weights[:len(removed_edges)]


def baseline(edge_type, P = 0.25, option = "CN"):
    """
    Run baseline function on a specific edge type.
    :param edge_type: the edge type string
    :param P: the edge removal probability
    :param option: the metric run
    :return: the number of edges predicted and the number of edges predicted correctly
    """
    print("Running baseline for", edge_type, "with function", option, "and P =", P)

    node1_type, node2_type = edge_type.split("_")[0], edge_type.split("_")[1]

    F = make_copy(G)
    F, removed_edges = remove_edges(F, edge_type, P)

    type1_nodes, type2_nodes = get_nodes(F, node1_type, node2_type)

    if option == "CN":
        removed_edge_weights = get_CN_values(F, type1_nodes, type2_nodes)
    elif option == "AA":
        node_pairs = get_node_pairs(F, type1_nodes, type2_nodes)
        generator = nx.adamic_adar_index(F, node_pairs)
        removed_edge_weights = get_generator_values(generator, node_pairs)
    elif option == "JC":
        node_pairs = get_node_pairs(F, type1_nodes, type2_nodes)
        generator = nx.jaccard_coefficient(F, node_pairs)
        removed_edge_weights = get_generator_values(generator, node_pairs)
    elif option == "RA":
        node_pairs = get_node_pairs(F, type1_nodes, type2_nodes)
        generator = nx.resource_allocation_index(F, node_pairs)
        removed_edge_weights = get_generator_values(generator, node_pairs)
    elif option == "PA":
        node_pairs = get_node_pairs(F, type1_nodes, type2_nodes)
        generator = nx.preferential_attachment(F, node_pairs)
        removed_edge_weights = get_generator_values(generator, node_pairs)

    removed_edge_weights = sorted(removed_edge_weights, key=lambda x: x[2], reverse=True)
    edges_predicted = removed_edge_weights[:len(removed_edges)]

    correct = analyze(edges_predicted, write = 1)

    return len(edges_predicted), correct


def random_walk_method(edge_type, bi_directed = False, runs = 1, P = 0.25, walks = 100, walk_len = 5):
    """
    Run our random walk method on a specific edge type.
    :param edge_type: the edge type string
    :param bi_directed: a binary T/F value that specifies whether we run random walks from only one node type
    :param runs: the number of times we run random walks without changing the removed edges
    :param P: the edge removal probability
    :param walks: the number of walks to be done from each node
    :param walk_len: the number of steps in each walk
    :return: IF runs == 1 then, the number of edges predicted and the number of edges predicted correctly
                          else, a list of fractions of edges predicted correctly vs edges predicted
    """
    if bi_directed:
        direction = "both directions"
    else:
        direction = "one direction"
    print("Running random walk method for", edge_type, "from", direction, "with P =", P, "and", "(" + str(walks) + "," + str(walk_len) + ")")

    node1_type, node2_type = edge_type.split("_")[0], edge_type.split("_")[1]

    F = make_copy(G)
    F, removed_edges = remove_edges(F, edge_type, P)

    type1_nodes, type2_nodes = get_nodes(F, node1_type, node2_type)

    if runs == 1:
        edges_predicted = run_walks(F, bi_directed, type1_nodes, type2_nodes, walks, walk_len, removed_edges)
        correct = analyze(edges_predicted, write=1)

        return len(edges_predicted), correct
    else:
        results = []
        while runs:
            edges_predicted = run_walks(F, bi_directed, type1_nodes, type2_nodes, walks, walk_len, removed_edges)
            correct = analyze(edges_predicted, write=1)
            results.append(correct / len(removed_edges))
            runs -= 1

        return results


def run_baseline_methods(edge_type, count = 1):
    """
    Run all baseline methods multiple times. This creates a file.
    :param edge_type: the edge type string
    :param count: the number of times each method is run
    :return: N/A
    """
    Ps = [0.1, 0.25, 0.5]
    options = ["CN", "AA", "JC", "RA", "PA"]

    file = open(edge_type + "_nx.txt", "w")

    for option in options:
        for P in Ps:
            file.write(option + "_" + str(P) + "\n")
            results = []
            for i in range(count):
                t, c = baseline(edge_type, P = P, option = option)
                results.append(c / t)
            file.write(str(np.mean(results)) + " " + str(np.std(results)) + "\n")

    file.close()

def run_our_methods_D(edge_type, count = 2):
    """
    Run our methods multiple times, using different removed edges each time. This creates a file.
    :param edge_type: the edge type string
    :param count: the number of times each method is run
    :return: N/A
    """
    Ps = [0.1, 0.25, 0.5]
    file = open(edge_type + "D.txt", "w")

    option = "1d1005"
    for P in Ps:
        file.write(option + "_" + str(P) + "\n")
        results = []
        for i in range(count):
            t, c = random_walk_method(edge_type, bi_directed=False, P = P, walks=100, walk_len=5)
            results.append(c / t)
        file.write(str(np.mean(results)) + " " + str(np.std(results)) + "\n")

    option = "1d2520"
    for P in Ps:
        file.write(option + "_" + str(P) + "\n")
        results = []
        for i in range(count):
            t, c = random_walk_method(edge_type, bi_directed=False, P = P, walks=25, walk_len=20)
            results.append(c / t)
        file.write(str(np.mean(results)) + " " + str(np.std(results)) + "\n")

    option = "2d1005"
    for P in Ps:
        file.write(option + "_" + str(P) + "\n")
        results = []
        for i in range(count):
            t, c = random_walk_method(edge_type, bi_directed=True, P = P, walks=100, walk_len=5)
            results.append(c / t)
        file.write(str(np.mean(results)) + " " + str(np.std(results)) + "\n")

    option = "2d2520"
    for P in Ps:
        file.write(option + "_" + str(P) + "\n")
        results = []
        for i in range(count):
            t, c = random_walk_method(edge_type, bi_directed=True, P = P, walks=25, walk_len=20)
            results.append(c / t)
        file.write(str(np.mean(results)) + " " + str(np.std(results)) + "\n")

    file.close()

def run_our_methods_S(edge_type, count = 2):
    """
    Run our methods multiple times, using the same removed edges each time. This creates a file.
    :param edge_type: the edge type string
    :param count: the number of times each method is run
    :return: N/A
    """
    Ps = [0.1, 0.25, 0.5]
    file = open(edge_type + "S.txt", "w")

    option = "1d1005"
    for P in Ps:
        file.write(option + "_" + str(P) + "\n")
        results = random_walk_method(edge_type, bi_directed=False, runs = count, P = P, walks=25, walk_len=20)
        file.write(str(np.mean(results)) + " " + str(np.std(results)) + "\n")

    option = "1d2520"
    for P in Ps:
        file.write(option + "_" + str(P) + "\n")
        results = random_walk_method(edge_type, bi_directed=False, runs = count, P=P, walks=25, walk_len=20)
        file.write(str(np.mean(results)) + " " + str(np.std(results)) + "\n")

    option = "2d1005"
    for P in Ps:
        file.write(option + "_" + str(P) + "\n")
        results = random_walk_method(edge_type, bi_directed=True, runs = count, P=P, walks=100, walk_len=5)
        file.write(str(np.mean(results)) + " " + str(np.std(results)) + "\n")

    option = "2d2520"
    for P in Ps:
        file.write(option + "_" + str(P) + "\n")
        results = random_walk_method(edge_type, bi_directed=True, runs = count, P=P, walks=25, walk_len=20)
        file.write(str(np.mean(results)) + " " + str(np.std(results)) + "\n")

    file.close()


G = get_graph()
node_categories = get_node_info()
get_stats(write = 1)

edge_type = "company_politician"
edge_type = corrected(edge_type)

# run a specific baseline method
baseline(edge_type, P = 0.25, option = "AA")

# run a specific version of our method
random_walk_method(edge_type, bi_directed = True, P = 0.25, walks = 100, walk_len = 5)

# run methods multiple times
run_baseline_methods(edge_type, count = 1)
run_our_methods_D(edge_type, count = 2) # count parameter in run_our_methods_D CANNOT be 1
run_our_methods_S(edge_type, count = 2) # count parameter in run_our_methods_S CANNOT be 1

