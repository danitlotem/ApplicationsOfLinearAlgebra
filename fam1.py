import time
import math
import random
import networkx as nx
import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
import pickle


def create_graph_family_one(n, q):
    # q = round(random.uniform(0, 1), 2) #probability to add edge to graph
    # g = nx.erdos_renyi_graph(n, q)  # create random graph

    g = nx.Graph()  # build empty graph

    for i in range(n):
        g.add_node(i)

    for i in range(n):
        for j in range(n):
            if np.random.choice(np.arange(1, 3), p=[q, 1 - q]) == 1:
                g.add_edge(i, j)

    return g


def create_graph_family_two(n):
    g = nx.Graph()

    for i in range(1, n + 1):
        g.add_node(i)

    nodes_and_probability = {}
    for i in range(1, n + 1):
        nodes_and_probability[i] = 0

    for i in range(1, n + 1):
        ## q=round(1/i,2)
        # q=round(1/math.sqrt(i),2)
        q = 1 / i
        nodes_and_probability[i] = q

    for i in range(1, n + 1):
        for j in range(1, n + 1):
            if np.random.choice(np.arange(1, 3), p=[nodes_and_probability[j], 1 - nodes_and_probability[j]]) == 1:
                g.add_edge(i, j)

    print("done building")
    return g


def normalize_d_arr(d, t):
    for i in range(0, len(d)):
        temp_lst = list(d[i])
        temp_lst[1] = temp_lst[1] / t

        d[i] = tuple(temp_lst)

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


# =============================================================================
#     sorted_d1 = sorted(dict1.items(), key=lambda x: (x[1], x[0]), reverse=True)
#     sorted_d2 = sorted(dict2.items(), key=lambda x: (x[1], x[0]), reverse=True)
#
#     lst1=[]
#     lst2=[]
#     for i in range(n):
#         lst1.append((sorted_d1[i][0],round(sorted_d1[i][1],2)))
#         lst2.append((sorted_d2[i][0],round(sorted_d2[i][1],2)))
#
#
#     for i in range(0,k):
#         if lst1[i] != lst2[i]:
#             return False
#
#     return True
# =============================================================================


def convert_dictionary_to_nparray(dictionary, n):
    lst = []
    for i in range(1, n):
        lst.append(dictionary[i])

    vector = np.array(lst)
    return vector


def page_rank_second_stop_condition(graph, N, prob, k, n):
    i = 1

    v = page_rank(graph, int(math.pow(2, i)), N, prob, n)
    u = page_rank(graph, int(math.pow(2, i + 1)), N, prob, n)

    while compare_k_elements_two_dicts(v, u, k, n) == False:
        i += 1
        v = page_rank(graph, int(math.pow(2, i)), N, prob, n)
        u = page_rank(graph, int(math.pow(2, i + 1)), N, prob, n)

        if compare_k_elements_two_dicts(v, u, k, n):
            break

        if (i > 18):
            print("i: " + str(i))

    new_u = np.zeros(n)
    new_v = np.zeros(n)

    j=0
    for j in range(n):
        new_u[j] = u[j][1]
        new_v[j] = v[j][1]

    new_vector = new_u - new_v
    norm_vector = norm(new_vector)
    print(norm_vector)


    return [i, v]


def page_rank(graph, t, N, prob, n):

    d = [(i, 0.0) for i in range(1, n+1)]
    d = np.array(d, dtype=[('ver', np.int16), ('d', np.float32)])

    curr_node = random.randint(1, n)

    for i in range(t):
        for j in range(N):

            if np.random.choice(np.arange(1, 3), p=[prob, 1 - prob]) == 1:
                curr_node = random.choice(list(graph.nodes))
            else:
                neighbors_list = list(graph.neighbors(curr_node))
                if len(neighbors_list) == 0:
                    curr_node = curr_node
                else:
                    curr_node = random.choice(neighbors_list)

        temp_lst = list(d[curr_node-1])
        temp_lst[1] = temp_lst[1] + 1
        d[curr_node-1] = tuple(temp_lst)

        curr_node = random.randint(1, n)

    normalized_d = normalize_d_arr(d, t)

    return normalized_d


def test():
    #n = int(math.pow(2, 9))
    n = int(math.pow(2, 12))

    #graph_1 = create_graph_family_one(n,1/math.pow(2, 16))
    graph_1 = pickle.load(open(r'C:/Users/danit/Lini/family2graph1.txt', 'rb'))
    #graph_1 = pickle.load(open(r'C:/Users/danit/Lini/graph2.txt', 'rb'))

    print("GRAPH CREATED")
    p = [1 / 2, 1 / 4, 1 / 8, 1 / 16, 1 / 32]

    k = [2, 4, 8, 16, 32]
    # p = [1 / 2]
    # k = [2]
    #N = 8
    graphes = []
    for k_elem in k:
        for p_elem in p:
            N = int(1 / p_elem)
            res_from_page_rank = page_rank_second_stop_condition(graph_1, N, p_elem, k_elem, n)
            t_res_graph = res_from_page_rank[0]
            if (p_elem == 1 / 32 and k_elem == 32):
                print("creating graph")
                d_array = res_from_page_rank[1]
                d_array_new = convert_dictionary_to_nparray(d_array, n)
                nodes_lst = [*range(1, n, 1)]
                plt.bar(nodes_lst, d_array_new, label="Data1")
                plt.legend()

            print("number of edges:" + str(graph_1.number_of_edges()))
            print("p:" + str(p_elem) + ", N:" + str(N) + ", k:" + str(k_elem) + ", t: 2^" + str(t_res_graph))
        print("\n")


if __name__ == '__main__':
    start = time.time()
    test()
    end = time.time()
    print(end - start)