import time
import math
import random
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import norm
import pickle


def calculate_norma_of_vector(vector1, vector2,n):
    new_v1=np.zeros(n)
    new_v2 = np.zeros(n)

    for i in range(n):
        new_v1[i]=vector1[i][1]
        new_v2[i] = vector2[i][1]

    new_vector = new_v1 - new_v2
    norm_vector = norm(new_vector)

    return norm_vector


def page_rank_first_stop_condition(graph, N, prob, epsilon, n):
    i = 1

    while True:
        v = page_rank(graph, int(math.pow(2, i)), N, prob, n)
        u = page_rank(graph, int(math.pow(2, i - 1)), N, prob, n)

        if calculate_norma_of_vector(v, u,n) < epsilon:
            break;
        i += 1

    new_k = compare_k_elements_two_dicts(v, u, n)
    print("k:"+str(new_k))

    return i


# normalize page rank arr (d array)
def normalize_d_arr(d, t):
    for i in range(0, len(d)):
        temp_lst = list(d[i])
        temp_lst[1] = temp_lst[1] / t

        d[i] = tuple(temp_lst)

    return d


def compare_k_elements_two_dicts(dict1, dict2, n):
    sorted_d1 = np.sort(dict1, order='d', kind='mergesort')[::-1]
    sorted_d2 = np.sort(dict2, order='d', kind='mergesort')[::-1]

    for i in range(n):
        if sorted_d1[i][0]!=sorted_d2[i][0]:
            return i

    return n



def page_rank(graph, t, N, prob, n):


    d = [(i, 0.0) for i in range(1,n+1)]
    d = np.array(d, dtype=[('ver', np.int16), ('d', np.float32)])

    curr_node = random.randint(1, n)  # choosing random node for start
    # d[curr_node] = d[curr_node] + 1

    for i in range(t):
        for j in range(N):
            neighbors_list = list(graph.neighbors(curr_node))

            if np.random.choice(np.arange(1, 3), p=[prob, 1 - prob]) == 1:
                curr_node = random.choice(list(graph.nodes))
            else:
                if len(neighbors_list) == 0:  # if there are no neighbours, stay in current node
                    curr_node = curr_node
                else:  # if there are neighbours, we choose random node with probability of p or random neighbour with probability of 1-p
                    curr_node = random.choice(neighbors_list)

        temp_lst = list(d[curr_node-1])
        temp_lst[1] = temp_lst[1] + 1
        d[curr_node-1] = tuple(temp_lst)

        curr_node = random.randint(1, n )

    # return normalized page rank arr
    normalized_d = normalize_d_arr(d, t)

    return normalized_d


def test():
    n = int(math.pow(2, 12))
    # n = int(math.pow(12, 2))

    graph_1 = pickle.load(open(r'C:/Users/danit/Lini/family2graph1.txt', 'rb'))
    # graph_2 = pickle.load(open(r'C:/Users/shell/Documents/graph2.txt', 'rb'))
    # graph_3 = pickle.load(open(r'C:/Users/shell/Documents/graph3.txt', 'rb'))
    print('graph created')

    p = [1 / 2, 1 / 4, 1 / 8, 1 / 16, 1 / 32]
    epsilon = [1 / 2, 1 / 4, 1 / 8, 1 / 16, 1 / 32]

    for epsilon_elem in epsilon:
        for p_elem in p:
            N = int(1 / p_elem)
            t_res_graph1 = page_rank_first_stop_condition(graph_1, N, p_elem, epsilon_elem, n)
            print("number of edges:" + str(graph_1.number_of_edges()))
            print(
                "p:" + str(p_elem) + ", N:" + str(N) + ", epsilon:" + str(epsilon_elem) + ", t: 2^" + str(t_res_graph1))
        print("\n")


if __name__ == '__main__':
    start = time.time()
    test()
    end = time.time()
    print(end - start)



