#!/usr/bin/env python
# encoding: utf-8  

""" 
@author: zzz_jq
@contact: zhuangjq@stu.xmu.edu.cn
@software: PyCharm 
@file: nn_pred.py 
@create: 2021/5/22 18:42 
"""
import os
import random
import numpy as np
import pandas as pd
import torch
from evaluation import eval_regression
from check_order import eval_ranking
from data_process_text import load_all_code_ids
from openpyxl import load_workbook

workload_dict = {
    "Spark ConnectedComponent Application": "CC",
    "DecisionTree classification Example": "DT",
    "Spark KMeans Example": "KM",
    "LinerRegressionApp Example": "LiR",
    "LogisticRegressionApp Example": "LoR",
    "Spark LabelPropagation Application": "LP",
    "MFApp Example": "MF",
    "Spark PCA Example": "PCA",
    "Spark PregelOperation Application": "PO",
    "Spark PageRank Application": "PR",
    "Spark StronglyConnectedComponent Application": "SCC",
    "Spark ShortestPath Application": "SP",
    "Spark SVDPlusPlus Application": "SVD",
    "SVM Classifier Example": "SVM",
    "Spark TriangleCount Application": "TC",
    "TeraSort": "TS"
}



#2024-12-5 Evaluate the SMLM model
def eval_workload_mixed(df, scaler,ws2adj,ws2nodes,model):
    # Normalize the data
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    numeric_cols = numeric_cols.drop(['Duration', 'stage_id']) # Exclude 'Duration' from normalization
    df[numeric_cols] = scaler.transform(df[numeric_cols].fillna(0))   
    df['Duration'] = df['Duration'].apply(lambda x: x/1000)

    data_rows = df.head(1)['rows'].tolist()[0]
    groups = df.groupby('AppId')
    all_target = []
    all_pred = []
    m=0
    xw = []
    xs=[]
    NODES = []
    ADJS = []
    ys=[]
    all_target=[]
    for app_id, group in groups:
        nodes_vec = []
        adj_vec = []
        #print(app_id)
        if group.head(1)['rows'].tolist()[0] != data_rows:
            continue
        workload, stage_id = group['AppName'].apply(lambda x: workload_dict[x]), group['stage_id']

        for w, s in zip(workload, stage_id):
            try:
                nodes_vec.append(ws2nodes[w][str(s)])
            except:
                nodes_vec.append(np.zeros(64, dtype=float))
            try:
                adj_vec.append(ws2adj[w][str(s)])
            except:
                adj_vec.append(np.zeros([64, 64], dtype=float))

        stage_feature = group[['stage_id', 'input', 'output', 'read', 'write']].values
        workload_feature = group.drop(['stage_id', 'duration', 'input', 'output', 'read', 'write','AppId', 'AppName', 'Duration','code'], axis=1).values
        # 只保留第一行数据
        workload_features_first_row = workload_feature[0]
        nodes_vec = np.stack(nodes_vec)
        adj_vec = np.stack(adj_vec)
        total_y = group['Duration'].tolist()[0]
        #X = group.drop(['AppId', 'AppName', 'Duration', 'code', 'duration', 'stage_id'], axis=1).values
        # workload_target = Y.sum()
        workload_target = total_y
        xw.append(workload_features_first_row)
        xs.append(stage_feature)
        NODES.append(nodes_vec)
        ADJS.append(adj_vec)
        all_target.append(workload_target)
        ys.append(group['duration'].values)
        m=m+1

        if m == 100:
            break

    xw=torch.FloatTensor(xw)

    #if torch.cuda.is_available():
      #  xw = xw.cuda()
       # model = model.cuda()
    
    workload_pred,stage_loss = model.forward(xw,xs, NODES, ADJS,ys)

    #print('end')
    all_pred = workload_pred.cpu().detach().numpy().flatten()
    all_target = np.array(all_target)
    #print('训练一次')
    res = evl(np.array(all_target), np.array(all_pred))
    #res = evl(all_target.detach().numpy(), all_pred.detach().numpy())
    return res    





def evl(all_target, all_pred):
    #输入运行时间和预测时间
    times = 10
    res1, res2 = np.array([0.0 for _ in range(5)]), np.array([0.0 for _ in range(3)])
    for _ in range(times):
        #print(_)
        #print(all_pred.shape[0])
        #将k改成200
        sample_indices = random.choices([i for i in range(all_pred.shape[0])], k=25)
        sample_pred = all_pred[sample_indices]
        sample_target = all_target[sample_indices]
        r1, r2 = eval_regression(sample_target, sample_pred), \
                 eval_ranking(sample_target, sample_pred)
        res1 += r1
        res2 += r2
    return res1 / times, res2 / times


def main(dataset_path,scaler,ws2adj,ws2nodes,model,all_code_dict):
    eval_result, ranking_result = [], []
    df_all = pd.read_csv(dataset_path, sep=',', low_memory=False)
    df_workloads = df_all.groupby('AppName')
    for w_name, w_df in df_workloads:
        #运行时间和预测时间
        #res, ranking_res = eval_workload_all(w_df, w_name,scaler,ws2adj,ws2nodes,model,all_code_dict)
        res,ranking_res=eval_workload_mixed(w_df, scaler,ws2adj,ws2nodes,model)
        eval_result.append(res)
        ranking_result.append(ranking_res)
     #mae, rmse, hr, ndcg, mrr
    print("regression eval: ")
    print(np.mean(np.array(eval_result), axis=0))
    print("ranking eval: ")
    print(np.mean(np.array(ranking_result), axis=0))
    return np.mean(np.array(eval_result), axis=0)[2],np.mean(np.array(eval_result), axis=0)[3]

def run():
    random.seed(2021)
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    all_code_dict = load_all_code_ids()

    #LITE
    #ws2nodes = torch.load('dag_data/old/ws2nodes.pth')
    #ws2adj = torch.load('dag_data/old/ws2adj.pth')

    #DSFlex
    ws2nodes = torch.load('dag_data/updated/ws2nodes.pth',weights_only=False)
    ws2adj = torch.load('dag_data/updated/ws2adj.pth',weights_only=False)

    # all-data
    #scaler = torch.load('model_save/model_save_1-4/all-data/Adj_Stage_M/scaler.pt')
    #data_path = 'dataset/dataset_4/test/test_data.csv'

    # never-cold-start
    scaler = torch.load('model_save/model_save_4/never-seen/DSFlex/scaler.pt',weights_only=False)
    data_path = 'dataset/dataset_4/train/train_data.csv'

    # never-warm-start
    # scaler = torch.load('model_save/model_save_4/all-data/scaler.pt')
    # data_path = 'dataset/dataset_4/test/test_data.csv'

    tuples1 = []
    tuples2 = []
    hr_5 = 0
    ndcg_5 = 0
    for i in range(0, (200)):
        print('epoch:' + str(i))
        if torch.cuda.is_available():

            # all-data
            #model = torch.load('model_save/model_save_stage/cnn_gcn_' + str(i) + '.pt')
            # never-warm-start
            #model = torch.load('model_save/model_save_1-4/all-data/Adj_Stage_M/cnn_gcn_' + str(i) + '.pt')
            # never-cold-start
            model = torch.load('model_save/model_save_4/never-seen/DSFlex/cnn_gcn_' + str(i) + '.pt',map_location='cpu',weights_only=False)
            # trans-learn
            # model = torch.load('model_save/model_save_3/trans-learn/G/trans_G' + str(i) + '.pt')
        else:
            # model = torch.load('model_save_1/cnn_gcn_'+str(i)+'.pt', map_location='cpu')
            model = torch.load('model_save_trans_3/trans' + str(i) + '.pt', map_location='cpu')
        hr, ndcg = main(data_path,scaler,ws2adj,ws2nodes,model,all_code_dict)
        t = (hr, i)
        k = (ndcg, i)
        tuples1.append(t)
        tuples2.append(k)

    tuples1.sort(key=lambda x: x[0])
    tuples1.reverse()
    tuples2.sort(key=lambda x: x[0])
    tuples2.reverse()
    for i in range(5):
        print('hr: ' + str(tuples1[i][0]) + 'model: ' + str(tuples1[i][1]))
        hr_5 = hr_5 + tuples1[i][0]
    hr_5 = hr_5 / 5
    print('hr@5:' + str(hr_5))
    for i in range(5):
        print('ndcg: ' + str(tuples2[i][0]) + 'model: ' + str(tuples2[i][1]))
        ndcg_5 = ndcg_5 + tuples2[i][0]
    ndcg_5 = ndcg_5 / 5
    print('ndcg@5:' + str(ndcg_5))


if __name__ == '__main__':
   run()





