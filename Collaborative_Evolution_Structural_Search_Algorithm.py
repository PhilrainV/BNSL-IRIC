import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import random
from tqdm import tqdm
import math
import time
from HCS_IRIC_main import *


class pop_obj:
    max_iter = 10
    pop_num = 50
    ambient_pressure = 0.2
    # -----------------------------------------------------------------------------------
    data = []
    do_data = []
    data_size = []
    nodes = []
    # -----------------------------------------------------------------------------------
    best_graphs = []
    best_adjas = []
    best_vecs = []
    best_scores = []
    # -----------------------------------------------------------------------------------
    pop_graph = []
    pop_adja = []
    pop_vec = []
    pop_score = []
    # -----------------------------------------------------------------------------------
    """
    输入：
    【data】DataFrame对象，含义是学生对某个知识点的掌握程度
    【do_data】DataFrame对象，含义是每一次作答的信息，每次作答的都是一个知识点
    【max_iter】最大迭代次数
    【pop_num】种群规模
    """
    def __init__(self, data, do_data, max_iter, pop_num, ambient_pressure):
        self.data = data
        self.data_size = len(data.columns)
        self.do_data = do_data
        self.max_iter = max_iter
        self.pop_num = pop_num
        self.nodes = data.columns
        self.ambient_pressure = ambient_pressure

        for i in tqdm(range(pop_num)):
            ind_graph, ind_adja, ind_vec = self.dag_random_init(data, 0.5)
            ind_graph, ind_adja, ind_vec, ind_score = self.dag_fix(ind_graph, ind_adja)
            self.pop_vec.append(ind_vec)
            self.pop_adja.append(ind_adja)
            self.pop_graph.append(ind_graph)
            self.pop_score.append(ANY_score(data, do_data, ind_graph))
        self.pop_rank()

        curr_best_graph = self.pop_graph[self.pop_num - 1].copy()
        self.best_graphs.append([curr_best_graph])
        self.best_adjas.append(self.pop_adja[self.pop_num - 1])
        self.best_vecs.append(self.pop_vec[self.pop_num - 1])
        self.best_scores.append(self.pop_score[self.pop_num - 1])
    # -----------------------------------------------------------------------------------
    def dag_random_init(self, data, gene_prob):
        """
        输入：
        【data】DataFrame对象，含义是学生对某个知识点的掌握程度
        【gene_prob】遍历每条可能的边时生成新边的概率，值在0到1之间

        输出：
        【dag】DiGraph对象，有向图
        【adja_matr】邻接矩阵，对于边i->j，存储为adja_matr(i,j)=1且adja_matr(j,i)=-1
        【ind_vec】决策变量，邻接矩阵adja_matr的主对角线以上元素按序组成的一维列表
        """
        dag = nx.DiGraph()
        dag.add_nodes_from(self.nodes)
        adja_matr = np.zeros((self.data_size, self.data_size))
        ind_vec = []

        for i in range(self.data_size):
            for j in range(self.data_size):
                if i == j:
                    continue
                else:
                    if np.random.rand() > 1 - gene_prob:
                        dag.add_edge(self.nodes[i - 1], self.nodes[j - 1])
                        adja_matr[i - 1, j - 1] = 1
                        adja_matr[j - 1, i - 1] = -1
        ind_vec = self.adja_to_vec(adja_matr)
        return dag, adja_matr, ind_vec
    # -----------------------------------------------------------------------------------
    def adja_to_graph(self, adja_matr):
        """
        输入：
        【adja_matr】邻接矩阵，对于边i->j，存储为adja_matr(i,j)=1且adja_matr(j,i)=-1

        输出：
        【ind_graph】有向图
        """
        # 初始化一个仅有节点的图
        ind_graph = nx.DiGraph()  # 创建空的有向图：DiGraph with 0 nodes and 0 edges
        ind_graph.add_nodes_from(self.nodes)  # 为有向图增加节点：DiGraph with 3 nodes and 0 edges

        # 在图中增加边
        for i in range(self.data_size):  # 遍历adja_matr中的每个元素：
            for j in range(self.data_size):
                if adja_matr[i, j] == 1:  # 如果发现了边i->j
                    ind_graph.add_edge(self.nodes[i - 1], self.nodes[j - 1])  # 有向图中增加边i->j
        return ind_graph

    # -----------------------------------------------------------------------------------
    def adja_to_vec(self, adja_matr):
        """
        输入：
        【adja_matr】邻接矩阵，对于边i->j，存储为adja_matr(i,j)=1且adja_matr(j,i)=-1

        输出：
        【ind_vec】决策变量，邻接矩阵adja_matr的主对角线以上元素按序组成的一维列表
        """
        n = len(adja_matr)
        ind_vec = []
        for i in range(n):
            for j in range(i + 1, n):
                if i < j:
                    ind_vec.append(adja_matr[i][j])
        return ind_vec
    # -----------------------------------------------------------------------------------、
    def vec_to_adja(self, ind_vec):
        """
        输入：
        【ind_vec】决策变量，邻接矩阵adja_matr的主对角线以上元素按序组成的一维列表

        输出：
        【adja_matr】邻接矩阵，对于边i->j，存储为adja_matr(i,j)=1且adja_matr(j,i)=-1
        """
        vec_len = len(ind_vec)
        n = int(((1 + pow((1 + 8 * vec_len), 0.5)) / 2))
        adja_matr = np.zeros([n, n])
        tmp = 0
        for i in range(1, n + 1):
            for j in range(i + 1, n + 1):
                # 如果元素在主对角线以上
                if i < j:
                    adja_matr[i - 1][j - 1] = ind_vec[tmp]  # 记录该元素
                    adja_matr[j - 1][i - 1] = -ind_vec[tmp]  # 记录对称位置的元素
                    tmp = tmp + 1  # 指针后移一位
        return adja_matr

    # -----------------------------------------------------------------------------------
    def pop_rank(self):
        num_len = len(self.pop_score)
        for j in range(num_len):
            sign = False
            for i in range(num_len - 1 - j):
                if self.pop_score[i] > self.pop_score[i + 1]:
                    self.pop_score[i], self.pop_score[i + 1] = self.pop_score[i + 1], self.pop_score[i]  # 交换种群评分
                    self.pop_graph[i], self.pop_graph[i + 1] = self.pop_graph[i + 1], self.pop_graph[i]  # 交换种群有向图
                    self.pop_adja[i], self.pop_adja[i + 1] = self.pop_adja[i + 1], self.pop_adja[i]  # 交换种群邻接矩阵
                    self.pop_vec[i], self.pop_vec[i + 1] = self.pop_vec[i + 1], self.pop_vec[i]  # 交换种群决策变量
                    sign = True
            if not sign:
                break
    # -----------------------------------------------------------------------------------
    def dag_fix(self, ind_graph, ind_adja):
        """
        函数作用：
        将一个有向图修复为一个有向无环图（若有环则随机删去环中的一条边）

        输入：
        【ind_graph】DiGraph对象，有向图
        【ind_adja】个体邻接矩阵

        输出：
        【ind_graph】修复后的DiGraph对象，有向图
        【ind_adja】修复后的个体邻接矩阵
        【ind_vec】修复后的个体决策变量
        【ind_score】更新后的个体评分
        """
        path_sequ = [[]]
        pointer = 0
        for i in range(self.data_size):
            path_sequ[0].append(i)
        random.shuffle(path_sequ[0])
        while True:
            if len(path_sequ[pointer]) != 0:
                tmp = path_sequ[pointer][0]
                next_node_set = []
                for j in range(self.data_size):
                    if ind_adja[tmp][j] == 1:
                        next_node_set.append(j)
                if len(next_node_set) != 0:
                    if path_sequ[0][0] in next_node_set:
                        ind_adja[tmp][path_sequ[0][0]] = 0
                        ind_adja[path_sequ[0][0]][tmp] = 0
                        ind_graph = self.adja_to_graph(ind_adja)
                    else:
                        random.shuffle(next_node_set)
                        path_sequ.append(next_node_set)
                        pointer = pointer + 1
                else:
                    for k in range(pointer + 1):
                        curr_pointer = pointer - k
                        if len(path_sequ[curr_pointer]) != 0:
                            del path_sequ[curr_pointer][0]
                        if len(path_sequ[curr_pointer]) != 0:
                            break
            else:
                del path_sequ[pointer]
                pointer = pointer - 1
            if len(path_sequ[0]) == 0:
                break
        ind_vec = self.adja_to_vec(ind_adja)
        ind_score = ANY_score(self.data, self.do_data, ind_graph)
        return ind_graph, ind_adja, ind_vec, ind_score

    # -----------------------------------------------------------------------------------
    def co_evolution(self):
        """
        输出：
        【best_dag】最优dag
        【best_score】最优dag的评分
        """
        for curr_iter in tqdm(range(0, self.max_iter)):
            covered_ind_num = max([1, int(self.pop_num * self.ambient_pressure)])
            self.pop_graph[0:covered_ind_num] = self.pop_graph[(self.pop_num - covered_ind_num):self.pop_num]
            self.pop_adja[0:covered_ind_num] = self.pop_adja[(self.pop_num - covered_ind_num):self.pop_num]
            self.pop_vec[0:covered_ind_num] = self.pop_vec[(self.pop_num - covered_ind_num):self.pop_num]
            self.pop_score[0:covered_ind_num] = self.pop_score[(self.pop_num - covered_ind_num):self.pop_num]

            for i in range(covered_ind_num, self.pop_num):
                changed_edges_num = math.ceil((1 - ((i + 1) / self.pop_num)) * self.pop_num)
                for changed_times in range(changed_edges_num):
                    start_node = np.random.randint(0, self.data_size - 1)
                    end_node = np.random.randint(0, self.data_size - 2)
                    if end_node >= start_node:
                        end_node = end_node + 1
                    edge_type = np.random.randint(-1, 1)
                    if edge_type == -1:
                        self.pop_adja[i][start_node][end_node] = -1
                        self.pop_adja[i][end_node][start_node] = 1
                        self.pop_graph[i] = self.adja_to_graph(self.pop_adja[i])
                    if edge_type == 0:
                        self.pop_adja[i][start_node][end_node] = 0
                        self.pop_adja[i][end_node][start_node] = 0
                        self.pop_graph[i] = self.adja_to_graph(self.pop_adja[i])
                    if edge_type == 1:
                        self.pop_adja[i][start_node][end_node] = 1
                        self.pop_adja[i][end_node][start_node] = -1
                        self.pop_graph[i] = self.adja_to_graph(self.pop_adja[i])
                ind_graph, ind_adja, ind_vec, ind_score = self.dag_fix(self.pop_graph[i], self.pop_adja[i])
                self.pop_graph[i] = ind_graph
                self.pop_adja[i] = ind_adja
                self.pop_vec[i] = ind_vec
                self.pop_score[i] = ind_score
            self.pop_rank()
            curr_best_graph = self.pop_graph[self.pop_num - 1].copy()
            self.best_graphs.append([curr_best_graph])
            self.best_adjas.append(self.pop_adja[self.pop_num - 1])
            self.best_vecs.append(self.pop_vec[self.pop_num - 1])
            self.best_scores.append(self.pop_score[self.pop_num - 1])
        best_dag = self.best_graphs[self.max_iter - 1]
        best_score = self.best_scores[self.max_iter - 1]
        return best_dag, best_score
    # -----------------------------------------------------------------------------------

# -----------------------------------------------------------------------------------
if __name__ == "__main__":
    # 定义评价函数
    def ANY_score(data, do_data, dag):
        """
        函数作用：
        评价一个有向无环图

        输入：
        【data】DataFrame对象，含义是学生对某个知识点的掌握程度
        【do_data】DataFrame对象，含义是每一次作答的信息，每次作答的都是一个知识点
        【dag】DiGraph对象，有向图

        输出：
        【score】分数，越大越好
        """
        score = random.uniform(0, 1.0)
        return score
    # -----------------------------------------------------------------------------------
    max_iter = 10  # 最大迭代次数
    pop_num = 50  # 种群规模
    ambient_pressure = 0.2  # 生存压力
    do_data = pd.DataFrame(
        {
            "student_nm": [121, 25, 364],
            "knowledge_id": ['knowledge_component_1', 'knowledge_component_2', 'knowledge_component_3'],
            "probability_id": [0.9582051, 0.9582051, 0.9582051]
        }
    )
    """
    结果为：
       student_nm          knowledge_id        probability_id
    0         121    knowledge_component_1        0.958205
    1          25    knowledge_component_2        0.958205
    2         364    knowledge_component_3        0.958205
    """
    # Data数据载入到 DataFrame 对象
    data = pd.DataFrame(
        {
            "K1": ['[1.0,, 0.0, 0.0, 0.0, 0.276111]', '[1.0,, 0.0, 0.0, 0.0, 0.276111]', '[1.0,, 0.0, 0.0, 0.0, 0.276111]'],
            "K2": ['[1.0,, 0.0, 0.0, 0.0, 0.276111]', '[1.0,, 0.0, 0.0, 0.0, 0.276111]', '[1.0,, 0.0, 0.0, 0.0, 0.276111]'],
            "K3": ['[1.0,, 0.0, 0.0, 0.0, 0.276111]', '[1.0,, 0.0, 0.0, 0.0, 0.276111]', '[1.0,, 0.0, 0.0, 0.0, 0.276111]']
        }
    )
    """
    结果为：
                                   K1                               K2  \
    0  [1.0,, 0.0, 0.0, 0.0, 0.276111]  [1.0,, 0.0, 0.0, 0.0, 0.276111]   
    1  [1.0,, 0.0, 0.0, 0.0, 0.276111]  [1.0,, 0.0, 0.0, 0.0, 0.276111]   
    2  [1.0,, 0.0, 0.0, 0.0, 0.276111]  [1.0,, 0.0, 0.0, 0.0, 0.276111]   
    
                                    K3  
    0  [1.0,, 0.0, 0.0, 0.0, 0.276111]  
    1  [1.0,, 0.0, 0.0, 0.0, 0.276111]  
    2  [1.0,, 0.0, 0.0, 0.0, 0.276111] 
    """
    # -----------------------------------------------------------------------------------

    pop = pop_obj(data, do_data, max_iter, pop_num, ambient_pressure)
    best_dag, best_score = pop.co_evolution()
    print('【main】所找到的最优个体的得分best_score={0}，决策变量best_vec={1},'.format(best_score,
                                                                                    pop.best_vecs[pop.max_iter - 1]))
    print('邻接矩阵best_adja=\n{0}，有向无环图best_dag为：'.format(pop.best_adjas[pop.max_iter - 1]))
    nx.draw_networkx(best_dag[0])

    print('接下来，绘制最优分数随迭代次数增长得到的进化曲线：')
    x = range(max_iter + 1)
    y = pop.best_scores
    print(pop.best_scores)

    print(best_dag[0])
    plt.figure(figsize=(20, 8), dpi=300)  # 1.创建画布
    plt.plot(x, y)  # 2.绘制图像
    plt.show()  # 3. 显示图像
