import pandas as pd
import numpy as np
from scipy.optimize import minimize
from dowhy import CausalModel
from tqdm import tqdm
from pyspark.sql import SparkSession
from pyspark import SparkConf
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

student_nm = 'user_id'
knowledge_id = 'exercise'
response_id = 'correct'
difficulty_id = 'difficulty'
probability_id = 'probability_id'

def count_calls(func):
    count = 0
    def wrapper(*args, **kwargs):
        nonlocal count
        count += 1
        print(f"`likelihood_function` 已被调用 {count} 次")
        return func(*args, **kwargs)
    return wrapper

def IRT_theta(do_data):
    global student_nm, knowledge_id, response_id, difficulty_id, probability_id

    data=do_data.copy()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_items = len(data[knowledge_id].unique())
    num_students = len(data[student_nm].unique())
    initial_ability = torch.zeros(num_students, requires_grad=True, device=device)
    data[knowledge_id] = pd.Categorical(data[knowledge_id], categories=data[knowledge_id].unique()).codes.astype(
        np.int64)
    data[student_nm] = pd.Categorical(data[student_nm], categories=data[student_nm].unique()).codes.astype(np.int64)
    data[response_id] = data[response_id].astype(np.float32)
    data[difficulty_id] = data[difficulty_id].astype(np.float32)
    knowledge_tensor = torch.tensor(data[knowledge_id].values, dtype=torch.int64).to(device)
    student_tensor = torch.tensor(data[student_nm].values, dtype=torch.int64).to(device)
    response_tensor = torch.tensor(data[response_id].values, dtype=torch.float32).to(device)
    difficulty_tensor = torch.tensor(data[difficulty_id].values, dtype=torch.float32).to(device)

    D = 1.702

    def likelihood_function(ability, knowledge, student, response, difficulty):
        theta_minus_b = ability[student] - difficulty
        probability = 1 / (1 + torch.exp(-D * theta_minus_b))
        log_likelihood = response * torch.log(probability) + (1 - response) * torch.log(1 - probability)
        return -log_likelihood.sum()

    dataset = TensorDataset(knowledge_tensor, student_tensor, response_tensor, difficulty_tensor)
    dataloader = DataLoader(dataset, batch_size=1024, shuffle=True)
    optimizer = torch.optim.Adam([initial_ability], lr=0.001)

    for epoch in tqdm(range(10)):
        for batch_knowledge, batch_student, batch_response, batch_difficulty in dataloader:
            optimizer.zero_grad()

            loss = likelihood_function(initial_ability, batch_knowledge, batch_student, batch_response,
                                       batch_difficulty)
            loss.backward()
            optimizer.step()
    initial_ability = torch.tensor(initial_ability).cuda()
    student_ids = data[student_nm].unique().tolist()

    theta = {student_id: initial_ability[i].item() for i, student_id in
             tqdm(enumerate(student_ids), total=len(student_ids), desc="Processing")}
    theta_cpu = {key: val for key, val in theta.items()}
    return theta_cpu

def IRT_P(data, theta):
    global student_nm, knowledge_id, response_id, difficulty_id, probability_id
    data_grouped = data.groupby([student_nm, knowledge_id], as_index=False)
    num_rows = sum(len(data_grouped.get_group(group)) for group in data_grouped.groups)
    col_names = ['student_nm', 'knowledge_id', 'probability_id']
    result = pd.DataFrame(index=np.arange(num_rows), columns=col_names)
    index = 0

    D = 1.702

    for group, subset in tqdm(data_grouped, total=len(data_grouped), desc="Students"):
        student, knowledge = group
        difficulty = subset[difficulty_id].values
        theta_student = theta[student]
        probability = 1 / (1 + np.exp(-D * (theta_student - difficulty)))
        len_prob = len(probability)
        result.iloc[index:index + len_prob] = np.column_stack(
            ([student] * len_prob, [knowledge] * len_prob, probability)
        )
        index += len_prob
    return result

def do(data, variable, parents):
    global student_nm, knowledge_id, response, difficulty, probability_id
    data = data.pivot_table(index="student_nm", columns="knowledge_id", values="probability_id")
    data.fillna(0, inplace=True)
    #data需要处理成知识点对应的不同学生的作答正确概率
    #            KC_1   KC_2   KC_3   KC_4
    # student_1   0.9    0.8    0.7    0.6
    # student_2   0.3    0.4    0.1    0.2
    # student_3   0.5    0.6    0.9    0.8

    model = CausalModel(
        data=data,
        treatment=parents,
        outcome=variable,
        common_causes=[element for element in data.columns if element not in [parents, variable]]
    )
    identified_estimand = model.identify_effect()

    #倾向性得分（离散）# Propensity Score Weighting（倾向性得分加权）,Propensity Score Matching（倾向性得分匹配）
    # estimate = model.estimate_effect(identified_estimand, method_name="backdoor.propensity_score_weighting")
    # estimate = model.estimate_effect(identified_estimand, method_name="backdoor.propensity_score_matching")
    # #线性回归（连续）
    estimate = model.estimate_effect(identified_estimand, method_name="backdoor.linear_regression")

    # #广义线性模型（Generalized Linear Model，GLM）用于建立与预测非正态响应变量的相关性
    # from statsmodels.api import families
    # method_params = {
    #     "glm_family": families.Binomial()
    # }
    # estimate = model.estimate_effect(identified_estimand, method_name="backdoor.generalized_linear_model",method_params=method_params)

    #回归不连续性方法 Regression Discontinuity
    # estimate = model.estimate_effect(identified_estimand, method_name="iv.regression_discontinuity",
    #                 method_params={'rd_variable_name':'response',
    #                    'rd_threshold_value':0.5,
    #                    'rd_bandwidth': 0.15})

    # #工具变量法
    # identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)
    # estimate = model.estimate_effect(identified_estimand,
    #                                  method_name="iv.instrumental_variable",
    #                                  test_significance=True)
    return estimate.value
