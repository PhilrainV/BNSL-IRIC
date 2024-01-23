import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.stats import entropy
from do import do, IRT_theta, IRT_P
from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['SimHei']
from tqdm import tqdm
from Collaborative_Evolution_Structural_Search_Algorithm import *
import torch
from scipy.sparse import csr_matrix
import ast


def calculate_mutual_information(matrix1, matrix2):
    from ast import literal_eval
    matrix1 = [literal_eval(elem) for elem in matrix1]
    matrix2 = [literal_eval(elem) for elem in matrix2]
    matrix1 = np.array(matrix1)
    matrix2 = np.array(matrix2)
    prob_matrix1 = matrix1 / np.sum(matrix1)
    prob_matrix2 = matrix2 / np.sum(matrix2)
    batch_size = 5000000
    mutual_info_sum = 0
    for i in range(0, prob_matrix1.shape[0], batch_size):
        batch1 = prob_matrix1[i:i + batch_size]
        batch2 = prob_matrix2[i:i + batch_size]
        batch1_torch = torch.tensor(batch1, dtype=torch.float32).cuda()
        batch2_torch = torch.tensor(batch2, dtype=torch.float32).cuda()
        batch1_torch.clamp_(min=1e-9)
        batch2_torch.clamp_(min=1e-9)
        log_ratio = torch.log(batch2_torch / batch1_torch)
        mutual_info = torch.sum(batch1_torch * log_ratio)
        mutual_info_sum += mutual_info.cpu().numpy()
    return mutual_info_sum

def calculate_likelihood(x):
    from ast import literal_eval
    x = [literal_eval(elem) for elem in x]
    x = np.array(x)
    likelihood = 0
    for values in x:
        mean = np.mean(values)
        variance = np.var(values)
        likelihood += -0.5 * np.sum(np.log(2 * np.pi * variance) + ((values - mean)**2) / variance)
    return likelihood

def tanh(x):
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

def IRIC_score(data, do_data, dag):
    n = len(data)
    p = len(dag.edges())
    score = 0
    edge_scores = []
    for node in dag.nodes():
        parents = list(dag.predecessors(node))

        if len(parents) == 0:
            score += calculate_likelihood(data[node])
        else:
            for parent in parents:
                MI_score = calculate_mutual_information(data[node], data[parent])
                casual_score = do(do_data, node, parent)
                w = 0.95
                score += (1-w)*tanh(MI_score) + w*tanh(casual_score)
                edge_score = (1-w)*abs(tanh(MI_score)) + w*abs(tanh(casual_score))
                edge_scores.append([parent, node,abs(MI_score),abs(casual_score), edge_score])
    return score - 0.01*(np.log(n - 1) * p)

def random_initialization(data):
    nodes = data.columns
    dag = nx.DiGraph()
    dag.add_nodes_from(nodes)
    for i in nodes:
        for j in nodes:
            if i == j:
                continue
            else:
                if np.random.rand() > 0.5:
                    dag.add_edge(i, j)
    return dag
# 爬山
def hill_climbing(data, do_data):
    nodes = data.columns
    dag =random_initialization(data)
    best_score = IRIC_score(data, do_data, dag)
    print(best_score)
    while True:
        improved = False
        for source in nodes:
            for target in nodes:
                if source != target and not dag.has_edge(source, target):
                    dag.add_edge(source, target)
                    new_score = IRIC_score(data, do_data, dag)
                    print('new_score',new_score)
                    print('best_score', best_score)
                    # 如果新的评分更好，则保留这条边
                    if new_score > best_score:
                        best_score = new_score
                        improved = True
                    else:
                        dag.remove_edge(source, target)
        if not improved:
            break
    return dag
#最大最小爬山
def MM_hill_climbing(data, do_data, max_iter=1):
    best_dag = None
    best_score = -np.inf
    for _ in tqdm(range(max_iter)):
        dag = random_initialization(data)
        score = IRIC_score(data, do_data, dag)
        while True:
            neighbors = []
            scores = []
            for node1, node2 in dag.edges:
                neighbor = dag.copy()
                neighbor.remove_edge(node1, node2)
                neighbors.append(neighbor)
            scores = [IRIC_score(data,do_data, X) for X in neighbors]
            if neighbors == []:
                break
            else:
                best_neighbor = neighbors[np.argmax(scores)]
                best_neighbor_score = scores[np.argmax(scores)]
            if best_neighbor_score > score:
                dag = best_neighbor
                score = best_neighbor_score
            else:
                break
        if score > best_score:
            best_dag = dag
            best_score = score
    print(best_score)
    return best_dag
def Other_scoreF(data):
    from pgmpy.estimators import HillClimbSearch, BicScore, AICScore, BDsScore, BDeuScore, K2Score
    est = HillClimbSearch(data)
    best_model = est.estimate(scoring_method=BDeuScore(data))
    dag = nx.DiGraph()
    dag.add_nodes_from(best_model.nodes())
    dag.add_edges_from(est.estimate(max_indegree=1).edges())
    return dag

if __name__ == "__main__":
    data = pd.read_csv("junyi_ProblemLog_original.csv")
    #do_data数据要求：
    # student_id    knowledge_id     P
    #     1             KC_1        0.1
    #     1             KC_2        0.3
    #     2             KC_3        0.2
    #     3             KC_3        0.7
    do_data = pd.read_csv('do_data.csv')
    print(do_data.head())
    #IM_data数据要求：
    # student_id    knowledge_id     feature_A  feature_B ...
    #     1             KC_1            a         ...
    #     1             KC_2            b         ...
    #     2             KC_3            c         ...
    #     3             KC_3            d         ...
    mi_data = pd.read_csv('IM_data.csv')
    knowledge_name_list = ['knowledge_component_1', 'knowledge_component_2','knowledge_component_3']
    random_10_percent = mi_data.sample(frac=0.01, replace=False, random_state=42)
    selected_indices = random_10_percent.index
    mi_data_10_percent = mi_data.loc[selected_indices,knowledge_name_list]
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    print('互信息数据:\n', mi_data_10_percent.head())
    selected_do_data = do_data[do_data['knowledge_id'].isin(knowledge_name_list)]

    # 执行普通图结构搜索算法搜索最佳结构
    #best_model = hill_climbing(data, do_data)
    #best_model = MM_hill_climbing(mi_data_10_percent, selected_do_data)
    best_model = Other_scoreF(mi_data_10_percent)

    #CEO-SS算法
    # max_iter = 1  # 最大迭代次数
    # pop_num = 20  # 种群规模
    # ambient_pressure = 0.2  # 生存压力
    # pop = pop_obj(mi_data_10_percent, selected_do_data, max_iter, pop_num, ambient_pressure)
    # best_dag, best_score = pop.co_evolution()
    # best_model = best_dag[0]

    print(best_model.edges())
    graph = nx.DiGraph()
    graph.add_nodes_from(best_model.nodes())
    graph.add_edges_from(best_model.edges())
    pos = nx.spring_layout(graph)
    nx.draw_networkx(graph, pos=pos, with_labels=True, node_color='lightblue', edge_color='gray', arrowsize=20)
    plt.show()
