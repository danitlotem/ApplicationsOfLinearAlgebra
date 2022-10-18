import time
import math
import random
import networkx as nx
import numpy as np
from numpy.linalg import norm
import pickle

def create_graph_family_one(n, q):
    g = nx.Graph()  # build empty graph

    for i in range(n):
        g.add_node(i)

    for i in range(n):
        for j in range(n):
            if np.random.choice(np.arange(1, 3), p=[q, 1 - q]) == 1:
                g.add_edge(i, j)

    print("build finished")
    return g


def normalize_d_arr(d, t):
    for i in range(len(d)):
        d[i] = d[i] / t
    return d


def get_index(elem):
    return elem[0]

def calculate_norma_of_vector(vector1, vector2):
    new_vector = vector1 - vector2
    norm_vector = norm(new_vector)

    return norm_vector

def compare_k_elements_two_dicts(dict1, dict2, k, n):
    sorted_d1 = np.sort(dict1, order='d', kind='mergesort')[::-1]
    sorted_d2 = np.sort(dict2, order='d', kind='mergesort')[::-1]

    if (np.array_equal(sorted_d1[:k]['ver'], sorted_d2[:k]['ver'])):
        print(sorted_d1[:k]['ver'])
        return True

    return False

def page_rank_second_stop_condition(graph, N, prob, k, n):
    i = 1

    while True:
        v = page_rank(graph, int(math.pow(2, i)), N, prob, n)
        u = page_rank(graph, int(math.pow(2, i + 1)), N, prob, n)

        if compare_k_elements_two_dicts(v, u, k,n):
            #epsilon = calculate_norma_of_vector(v.values(), u.values())
            #epsilon = calculate_norma_of_vector(v, u)
            #print("distance between vectors: "+str(epsilon)) #---------------------------------------------------!!!
            break

        i += 1

    return i


def page_rank(graph, t, N, prob, n):

    d = {}

    for j in range(n):
        d[j] = 0

    curr_node = random.randint(0, n - 1)

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
        curr_node = random.randint(0, n - 1)

    normalized_d = normalize_d_arr(d, t)

    return normalized_d


def test():
    n = int(math.pow(2, 12))

    #graph_1 = create_graph_family_one(n, 1 / (int(math.pow(2, 12))))
    #graph_1 = create_graph_family_one(n, 1/(int(math.pow(2, 9))))
    #graph_1 = create_graph_family_one(n, 1/(int(math.pow(2, 4))))

    graph_1 = pickle.load(open(r'C:/Users/danit/Lini/graph2.txt', 'rb'))

    print("GRAPH CREATED")
    p = [1 / 2, 1 / 4, 1 / 8, 1 / 16, 1 / 32]
    k = [2, 4, 8, 16, 32]

    for k_elem in k:
        for p_elem in p:
            N = int(1 / p_elem)
            t_res_graph = page_rank_second_stop_condition(graph_1, N, p_elem, k_elem, n)
            print("number of edges:" + str(graph_1.number_of_edges()))
            print("p:" + str(p_elem) + ", N:" + str(N) + ", k:" + str(k_elem)+", t: 2^" + str(t_res_graph))
        print("\n")


if __name__ == '__main__':
    start = time.time()
    test()
    end = time.time()
    print(end - start)