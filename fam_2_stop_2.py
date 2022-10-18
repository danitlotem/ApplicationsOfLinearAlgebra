import time
import math
import random
import networkx as nx
import numpy as np
import pickle

def create_graph_family_two(n):
    g = nx.DiGraph()

    for i in range(1, n + 1):
        g.add_node(i)

    nodes_and_probability = {}
    for i in range(1, n + 1):
        nodes_and_probability[i] = 0

    for i in range(1, n + 1):
        q=1/i
        nodes_and_probability[i] = q

    for i in range(1, n + 1):
        for j in range(1, n + 1):
            if np.random.choice(np.arange(1, 3), p=[nodes_and_probability[j], 1 - nodes_and_probability[j]]) == 1:
                g.add_edge(i, j)

    print("done building")
    return g


def normalize_d_arr(d, t):
    for i in range(1, len(d) + 1):
        d[i] = round(d[i] / t, 2)
    return d


def get_index(elem):
    return elem[0]


def compare_k_elements_two_dicts(dict1, dict2, k):
    sorted_d1 = sorted(dict1.items(), key=lambda x: (x[1], x[0]), reverse=True)
    sorted_d2 = sorted(dict2.items(), key=lambda x: (x[1], x[0]), reverse=True)

    for i in range(1, k + 1):
        if sorted_d1[i] != sorted_d2[i]:
            return False

    return True


def page_rank_second_stop_condition(graph, N, prob, k, n):
    i = 1

    while True:
        v = page_rank(graph, int(math.pow(2, i)), N, prob, n)
        u = page_rank(graph, int(math.pow(2, i + 1)), N, prob, n)

        if compare_k_elements_two_dicts(v, u, k):
            break

        i += 1

    return i


def page_rank(graph, t, N, prob, n):
    d = {}

    for j in range(1, n + 1):
        d[j] = 0

    curr_node = random.randint(1, n)

    for i in range(t):
        for j in range(N):
            neighbors_list = list(graph.neighbors(curr_node))

            if np.random.choice(np.arange(1, 3), p=[prob, 1 - prob]) == 1:
                curr_node = random.choice(list(graph.nodes))
            else:
                if len(neighbors_list) == 0:
                    curr_node = curr_node
                else:
                    curr_node = random.choice(neighbors_list)

        d[curr_node] = d[curr_node] + 1
        curr_node = random.randint(1, n)

    normalized_d = normalize_d_arr(d, t)

    return normalized_d


def test():
    n = int(math.pow(2, 12))
    # n = int(math.pow(2, 4))

    graph_1 = create_graph_family_two(n)
    #graph_1 = pickle.load(open(r'C:/Users/danit/Lini/family2test1.txt', 'rb'))


    print("GRAPH CREATED")
    p = [1 / 2, 1 / 4, 1 / 8, 1 / 16, 1 / 32]
    k = [2, 4, 8, 16, 32]

    for k_elem in k:
        for p_elem in p:
            N = int(1 / p_elem)
            t_res_graph = page_rank_second_stop_condition(graph_1, N, p_elem, k_elem, n)
            print("number of edges:" + str(graph_1.number_of_edges()))
            print("p:" + str(p_elem) + ", N:" + str(N) + ", k:" + str(k_elem) + ", t: 2^" + str(t_res_graph))
        print("\n")


if __name__ == '__main__':
    start = time.time()
    test()
    end = time.time()
    print(end - start)