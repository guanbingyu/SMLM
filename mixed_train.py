import time
import torch
from torch import nn
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from models import *
from my_dataset import SparkJobDataset, collate_fn
from config import Config
import os
import math
import nn_pred_1
from dataset_process import build_mixed_dataset
from models import SparkJobModel
from evaluation import eval_regression
from stage_pred import evl
import random
import warnings
warnings.filterwarnings("ignore")
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


#select_data_by_w 
def select_data_by_w(batch, num_samples=25):
    tensor_xw, tensor_y, tensor_xs, nodes, adj, y, w = batch
    selected_indices = []
    unique_w = torch.unique(w)
    for uw in unique_w:
        indices = (w == uw).nonzero(as_tuple=True)[0]
        if len(indices) > num_samples:
            indices = indices[:num_samples]
        selected_indices.extend(indices.tolist())
    selected_indices = torch.tensor(selected_indices)
    return (tensor_xw[selected_indices], tensor_y[selected_indices], tensor_xs[selected_indices], 
            nodes[selected_indices], adj[selected_indices], y[selected_indices], w[selected_indices])

def validate_on_validset(model, data_by_w, unique_WV):
    model.eval()
    eval_result, ranking_result = [], []
    selected_data = {'XWV': [], 'YV': [], 'XSV': [], 'NODESV': [], 'ADJSV': [], 'YSV': []}
    for uw in unique_WV:
        selected_data['XWV'] = torch.stack(data_by_w[uw]['XWV'])
        #selected_data['XWV'] = data_by_w[uw]['XWV']
        selected_data['YV'] = data_by_w[uw]['YV']
        selected_data['XSV'] = data_by_w[uw]['XSV']
        selected_data['NODESV'] = data_by_w[uw]['NODESV']
        selected_data['ADJSV'] = data_by_w[uw]['ADJSV']
        selected_data['YSV'] = data_by_w[uw]['YSV']

        #num_samples = min(2000, len(selected_data['XWV']))
        #indices = random.sample(range(len(selected_data['XWV'])), num_samples)
        #selected_data_sampled = {key: [value[i] for i in indices] for key, value in selected_data.items()}
        #y_pred, stage_loss = model.forward(selected_data_sampled['XWV'], selected_data_sampled['XSV'], selected_data_sampled['NODESV'], selected_data_sampled['ADJSV'], selected_data_sampled['YSV'])
        y_pred, stage_loss = model.forward(selected_data['XWV'], selected_data['XSV'], selected_data['NODESV'], selected_data['ADJSV'], selected_data['YSV'])
        predicted = y_pred.cpu().data
        valid_all_pred = predicted.numpy().tolist()
        valid_all_y = torch.stack(data_by_w[uw]['YV']).cpu().long().numpy()
        #valid_all_y=torch.stack(selected_data_sampled['YV']).cpu().long().numpy()
        # Avoid NAN on the validation set
        Rvalid_all_pred = np.array(valid_all_pred).flatten()
        res, ranking_res = evl(np.array(valid_all_y), np.array(Rvalid_all_pred))
        eval_result.append(res)
        ranking_result.append(ranking_res)
    print("regression eval: ")
    print(np.mean(np.array(eval_result), axis=0))
    print("ranking eval: ")
    print(np.mean(np.array(ranking_result), axis=0))

    return eval_result, ranking_result

#Select different training sets and validation sets according to the differences of cold_start and warm_start  
#cold_start 2025-3-12
#{'SVM', 'PCA', 'SCC', 'LoR', 'DT', 'LP', 'PR', 'MF', 'LiR', 'TC', 'KM', 'SP', 'CC', 'SVD', 'PO'} Select some of them as the training set and the others as the test set.  
def dataset_select(train_dataset, validate_dataset,train_and_validate=True,cold_start=False,trainset={},validset={}):
    if train_and_validate:
        if int(train_dataset) ==int(validate_dataset):
            W, S, CODE, NODES, ADJS, XS, XW, Y,ys = build_mixed_dataset(dataset_path='dataset/dataset_'+train_dataset+'/train/train_data.csv',
                                                                        scaler_save_path='model_save/model_save_'+train_dataset+'/all-data/DSFlex/scaler.pt', 
                                                                        conf_range_path='config_range/Spark_conf_range.xlsx')
            NODES = np.array(list(NODES.values()),dtype=object)
            ADJS = np.array(list(ADJS.values()),dtype=object)
            XS= np.array(list(XS.values()),dtype=object)
            XW = np.array(list(XW.values()))

            #XW = np.array(list(XW.values()),dtype=object)
            Y = np.array(list(Y.values()))
            ys1 = np.array(list(ys.values()),dtype=object)
            dataset = SparkJobDataset(XW, Y, XS, NODES, ADJS,ys, W)
            try:
                train_set, validate_set = random_split(dataset, [int(0.8 * len(dataset)), int(0.2 * len(dataset))])
            except:
                train_set, validate_set = random_split(dataset, [int(0.8 * len(dataset)), int(0.2 * len(dataset)) + 1])
        else:
            W1, S1, CODE1, NODES1, ADJS1, XS1, XW1, Y1,ys1 = build_mixed_dataset(dataset_path='dataset/dataset_'+train_dataset+'/train/train_data.csv',
                                                                        scaler_save_path='model_save/model_save_'+train_dataset+'/all-data/DSFlex/scaler.pt',  
                                                                        conf_range_path='config_range/Spark_conf_range.xlsx')
            NODES1 = np.array(list(NODES1.values()),dtype=object)
            ADJS1 = np.array(list(ADJS1.values()),dtype=object)
            XS1 = np.array(list(XS1.values()),dtype=object) 
            XW1 = np.array(list(XW1.values()))
            Y1 = np.array(list(Y1.values()))
            ys1 = np.array(list(ys1.values()),dtype=object)
            dataset1 = SparkJobDataset(XW1, Y1, XS1, NODES1, ADJS1,ys1, W1)

            W2, S2, CODE2, NODES2, ADJS2, XS2, XW2, Y2,ys2 = build_mixed_dataset(dataset_path='dataset/dataset_'+validate_dataset+'/train/train_data.csv',
                                                                        scaler_save_path='model_save/model_save_'+validate_dataset+'/all-data/DSFlex/scaler.pt',
                                                                        conf_range_path='config_range/Spark_conf_range.xlsx')
            NODES2 = np.array(list(NODES2.values()),dtype=object)
            ADJS2 = np.array(list(ADJS2.values()),dtype=object)
            XS2 = np.array(list(XS2.values()),dtype=object)
            XW2 = np.array(list(XW2.values()))
            Y2 = np.array(list(Y2.values()))
            ys2 = np.array(list(ys2.values()),dtype=object)
            dataset2 = SparkJobDataset(XW2, Y2, XS2, NODES2, ADJS2,ys2, W2)
            try:
                train_set, _ = random_split(dataset1, [int(0.8 * len(dataset1)), int(0.2 * len(dataset1))])
            except:
                train_set, _ = random_split(dataset1, [int(0.8 * len(dataset1)), int(0.2 * len(dataset1)) + 1])
            
            try:
                _, validate_set = random_split(dataset2, [int(0.8 * len(dataset2)), int(0.2 * len(dataset2))])
            except:
                _, validate_set = random_split(dataset2, [int(0.8 * len(dataset2)), int(0.2 * len(dataset2)) + 1])
    else:
        W, S, CODE, NODES, ADJS, XS, XW, Y,ys = build_mixed_dataset(dataset_path='dataset/dataset_'+validate_dataset+'/train/train_data.csv',
                                                                        scaler_save_path='model_save/model_save_'+validate_dataset+'/all-data/DSFlex/scaler.pt', 
                                                                        conf_range_path='config_range/Spark_conf_range.xlsx')
        NODES = np.array(list(NODES.values()),dtype=object)
        ADJS = np.array(list(ADJS.values()),dtype=object)
        XS= np.array(list(XS.values()),dtype=object)
        XW = np.array(list(XW.values()))
        Y = np.array(list(Y.values()))
        ys = np.array(list(ys.values()),dtype=object)
        dataset = SparkJobDataset(XW, Y, XS, NODES, ADJS,ys, W)
        try:
            train_set, validate_set = random_split(dataset, [int(0.8 * len(dataset)), int(0.2 * len(dataset))])
        except:
            train_set, validate_set = random_split(dataset, [int(0.8 * len(dataset)), int(0.2 * len(dataset)) + 1])
    #Extract the data from the training set  
    XWT=train_set.dataset.workload_feat
    YT=train_set.dataset.labels
    XST=train_set.dataset.stage_feats
    NODEST=train_set.dataset.dag_nodes
    ADJST=train_set.dataset.dag_adjs
    YST=train_set.dataset.ys
    WT=train_set.dataset.W      
    
    #Extract the data from the test set
    XWV=validate_set.dataset.workload_feat
    YV=validate_set.dataset.labels
    XSV=validate_set.dataset.stage_feats
    NODESV=validate_set.dataset.dag_nodes
    ADJSV=validate_set.dataset.dag_adjs
    YSV=validate_set.dataset.ys
    WV=validate_set.dataset.W

    # 找出WV中的不同值
    #WV = validate_set.dataset.W
    unique_WV = set(WV)
    print("Unique values in WV:", unique_WV)
    if cold_start:
        traindata_by_w = {uw: {'XWV': [], 'YV': [], 'XSV': [], 'NODESV': [], 'ADJSV': [], 'YSV': [],'WV':[] } for uw in unique_WV}
        for i, w in enumerate(WT):
            traindata_by_w[w]['XWV'].append(XWT[i])
            traindata_by_w[w]['YV'].append(YT[i])
            traindata_by_w[w]['XSV'].append(XST[i])
            traindata_by_w[w]['NODESV'].append(NODEST[i])
            traindata_by_w[w]['ADJSV'].append(ADJST[i])
            traindata_by_w[w]['YSV'].append(YST[i])
            traindata_by_w[w]['WV'].append(WT[i])

        # 选择特定的workload作为训练集
        selected_traindata = {'XWV': [], 'YV': [], 'XSV': [], 'NODESV': [], 'ADJSV': [], 'YSV': [],'WV':[]}
        for uw in trainset:
            selected_traindata['XWV'].extend(traindata_by_w[uw]['XWV'])
            selected_traindata['YV'].extend(traindata_by_w[uw]['YV'])
            selected_traindata['XSV'].extend(traindata_by_w[uw]['XSV'])
            selected_traindata['NODESV'].extend(traindata_by_w[uw]['NODESV'])
            selected_traindata['ADJSV'].extend(traindata_by_w[uw]['ADJSV'])
            selected_traindata['YSV'].extend(traindata_by_w[uw]['YSV'])
            selected_traindata['WV'].extend(traindata_by_w[uw]['WV'])
        print('trainset:')
        print(trainset)
        # 将累加后的数据转换为张量
        selected_traindata['XWV'] = torch.stack(selected_traindata['XWV'])
        traindataset=SparkJobDataset(selected_traindata['XWV'], selected_traindata['YV'],
                                      selected_traindata['XSV'], selected_traindata['NODESV'], selected_traindata['ADJSV'], selected_traindata['YSV'],selected_traindata['WV'])
        
    # 根据unique_WV中的不同值，得到对应的XWV, YV等
    data_by_w = {uw: {'XWV': [], 'YV': [], 'XSV': [], 'NODESV': [], 'ADJSV': [], 'YSV': []} for uw in unique_WV}
    for i, w in enumerate(WV):
        data_by_w[w]['XWV'].append(XWV[i])
        data_by_w[w]['YV'].append(YV[i])
        data_by_w[w]['XSV'].append(XSV[i])
        data_by_w[w]['NODESV'].append(NODESV[i])
        data_by_w[w]['ADJSV'].append(ADJSV[i])
        data_by_w[w]['YSV'].append(YSV[i])
    
    if cold_start:
        select_validdata = {uw: {'XWV': [], 'YV': [], 'XSV': [], 'NODESV': [], 'ADJSV': [], 'YSV': []} for uw in validset}
        for w in validset:
            select_validdata[w]['XWV'].extend(data_by_w[w]['XWV'])
            select_validdata[w]['YV'].extend(data_by_w[w]['YV'])    
            select_validdata[w]['XSV'].extend(data_by_w[w]['XSV'])
            select_validdata[w]['NODESV'].extend(data_by_w[w]['NODESV'])
            select_validdata[w]['ADJSV'].extend(data_by_w[w]['ADJSV'])
            select_validdata[w]['YSV'].extend(data_by_w[w]['YSV'])
        print('validset:')
        print(validset)
        return traindataset, select_validdata, validset

    return train_set, validate_set, data_by_w, unique_WV



    



#validate without train 2025-3-11
def validate_without_train(train_dataset,validate_dataset,epoch_number=200,cold_start=False,trainset={},validset={}):
    tuples1 = []
    tuples2 = []
    hr_5 = 0
    ndcg_5 = 0

    # 选择数据集
    if cold_start:
        _,data_by_w, unique_WV = dataset_select(train_dataset, validate_dataset,train_and_validate=False,cold_start=True,trainset=trainset,validset=validset)
    else:
        _, _, data_by_w, unique_WV = dataset_select(train_dataset, validate_dataset,train_and_validate=False)

    for i in range(0, epoch_number):
        start = time.time()
        print('epoch:' + str(i))
        if torch.cuda.is_available():

            # never-cold-start
            if cold_start:
                model = torch.load('model_save/model_save_'+train_dataset+'/never-seen/DSFlex/cnn_gcn_' + str(i) + '.pt',weights_only=False)      
            else:
                model = torch.load('model_save/model_save_'+train_dataset+'/all-data/DSFlex/cnn_gcn_' + str(i) + '.pt',weights_only=False)
        else:
            model = torch.load('model_save_trans_3/trans' + str(i) + '.pt', map_location='cpu')


        eval_result, _ = validate_on_validset(model, data_by_w, unique_WV)
        hr = np.mean(np.array(eval_result), axis=0)[2]
        ndcg = np.mean(np.array(eval_result), axis=0)[3]
        end = time.time()
        print("time cost: " + str(end - start))
        print("####################################################################")
        t = (hr, i)
        k = (ndcg, i)
        tuples1.append(t)
        tuples2.append(k)

    #get hr_5 and ndcg_5
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
    print('end')






def train_with_validate(BreakPoint=0,train_dataset='4',validate_dataset='4',cold_start=False,trainset={},validset={}):
    config = Config()
    #获得hr_5,ndcg_5
    tuples1 = []
    tuples2 = []
    hr_5 = 0
    ndcg_5 = 0
    #never-seen-experiment

    # 选择数据集
    if cold_start:
        train_set, data_by_w, unique_WV = dataset_select(train_dataset, validate_dataset,train_and_validate=True,cold_start=True,trainset=trainset,validset=validset)
    else:
        train_set, _, data_by_w, unique_WV = dataset_select(train_dataset, validate_dataset,train_and_validate=True)

    # 创建数据集和数据加载器
    #断点训练
    #model = torch.load('model_save/model_save_4/all-data/DSFlex/cnn_gcn_199.pt', weights_only=False)

    # 初始化模型
    model = SparkJobModel(workload_feat_dim=25, stage_feat_dim=105, hidden_dim=64)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)



    train_loader = DataLoader(train_set, batch_size=512, shuffle=True, collate_fn=collate_fn)
    #valid_loader = DataLoader(validate_set, batch_size=512, shuffle=True, collate_fn=collate_fn)
    #采用cuda加速
    if torch.cuda.is_available():
        print('use cuda')
        model = model.cuda()
        criterion = criterion.cuda()


    for epoch in range(config.epoch_num-BreakPoint):

        if (epoch+1) % config.lr_decay_epochs == 0:
            for g in optimizer.param_groups:
                g['lr'] = g['lr'] * config.lr_decay_ratio
        model.train()
        train_all_y = []
        train_all_pred = []
        for i, batch in enumerate(train_loader):
            start = time.time()
            #print(str(epoch) + ": " + str(i))
            tensor_xw,  tensor_y,tensor_xs, nodes, adj ,y,WT= batch
            if torch.cuda.is_available():
                #tensor_xw, tensor_y = tensor_xw.cuda() ,tensor_y.cuda()
                tensor_y = tensor_y.cuda()
                #nodes, adj = nodes.cuda(), adj.cuda()
            y_pred,stage_loss = model.forward(tensor_xw,tensor_xs, nodes, adj,y)
            workload_loss = criterion(y_pred, tensor_y.float().unsqueeze(1))
            total_loss = workload_loss + stage_loss
            # print('     loss: ' + str(loss.item()))
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            # train metrics
            predicted = y_pred.cpu().data
            train_all_pred.extend(predicted.numpy().tolist())
            train_all_y.extend(tensor_y.cpu().long().numpy())
            # break
        #避免预测值出现越界情况
        Rtrain_all_pred=np.array(train_all_pred).flatten()
        for i in range(len(Rtrain_all_pred)):
            #print(Rtrain_all_pred[i])
            #print(math.nan)
            if math.isnan(Rtrain_all_pred[i]):
                print('transform to 1')
                Rtrain_all_pred[i] = 100

        train_mae = mean_absolute_error(train_all_y, Rtrain_all_pred)
        #train_mae = mean_absolute_error(train_all_y, np.array(train_all_pred).flatten())
        train_rmse = np.sqrt(mean_squared_error(train_all_y, Rtrain_all_pred))
        print("epoch " + str(epoch) + ": train MAE: " + str(train_mae) + '; RMSE: ' + str(train_rmse))
        #never-seen-experiment
        if cold_start:
            torch.save(model, 'model_save/model_save_'+train_dataset+'/never-seen/DSFlex/cnn_gcn_' + str(epoch) + '.pt')
        else:
            torch.save(model, 'model_save/model_save_'+train_dataset+'/all-data/DSFlex/cnn_gcn_' + str(epoch) + '.pt')
  



        eval_result, _ = validate_on_validset(model, data_by_w, unique_WV)

        hr = np.mean(np.array(eval_result), axis=0)[2]
        ndcg = np.mean(np.array(eval_result), axis=0)[3]
        end = time.time()
        print("time cost: " + str(end - start))
        print("####################################################################")
        t = (hr, epoch)
        k = (ndcg, epoch)
        tuples1.append(t)
        tuples2.append(k)

    #get hr_5 and ndcg_5
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
    print('end')


def train_without_validate(BreakPoint=0,train_dataset='4',validate_dataset='4',cold_start=False,trainset={},validset={}):
    config = Config()
    #get hr_5,ndcg_5
    tuples1 = []
    tuples2 = []
    hr_5 = 0
    ndcg_5 = 0
    #never-seen-experiment

    # choose dataset
    if cold_start:
        train_set, data_by_w, unique_WV = dataset_select(train_dataset, validate_dataset,train_and_validate=True,cold_start=True,trainset=trainset,validset=validset)
    else:
        train_set, _, data_by_w, unique_WV = dataset_select(train_dataset, validate_dataset,train_and_validate=True)

    # Create datasets and data loaders
    #断Breakpoint training
    #model = torch.load('model_save/model_save_4/all-data/DSFlex/cnn_gcn_199.pt', weights_only=False)

    # Initialize the model
    model = SparkJobModel(workload_feat_dim=25, stage_feat_dim=105, hidden_dim=64)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.006)



    train_loader = DataLoader(train_set, batch_size=512, shuffle=True, collate_fn=collate_fn)
    #valid_loader = DataLoader(validate_set, batch_size=512, shuffle=True, collate_fn=collate_fn)
    #Adopt cuda acceleration
    if torch.cuda.is_available():
        print('use cuda')
        model = model.cuda()
        criterion = criterion.cuda()


    for epoch in range(config.epoch_num-BreakPoint):

        if (epoch+1) % config.lr_decay_epochs == 0:
            for g in optimizer.param_groups:
                g['lr'] = g['lr'] * config.lr_decay_ratio
        model.train()
        train_all_y = []
        train_all_pred = []
        for i, batch in enumerate(train_loader):
            start = time.time()
            #print(str(epoch) + ": " + str(i))
            tensor_xw,  tensor_y,tensor_xs, nodes, adj ,y,WT= batch
            if torch.cuda.is_available():
                #tensor_xw, tensor_y = tensor_xw.cuda() ,tensor_y.cuda()
                tensor_y = tensor_y.cuda()
                #nodes, adj = nodes.cuda(), adj.cuda()
            y_pred,stage_loss = model.forward(tensor_xw,tensor_xs, nodes, adj,y)
            workload_loss = criterion(y_pred, tensor_y.float().unsqueeze(1))
            total_loss = workload_loss + stage_loss
            # print('     loss: ' + str(loss.item()))
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            # train metrics
            predicted = y_pred.cpu().data
            train_all_pred.extend(predicted.numpy().tolist())
            train_all_y.extend(tensor_y.cpu().long().numpy())
            # break
        #Avoid the situation where the predicted values are out of bounds
        Rtrain_all_pred=np.array(train_all_pred).flatten()
        for i in range(len(Rtrain_all_pred)):
            #print(Rtrain_all_pred[i])
            #print(math.nan)
            if math.isnan(Rtrain_all_pred[i]):
                print('转换成1')
                Rtrain_all_pred[i] = 100

        train_mae = mean_absolute_error(train_all_y, Rtrain_all_pred)
        #train_mae = mean_absolute_error(train_all_y, np.array(train_all_pred).flatten())
        train_rmse = np.sqrt(mean_squared_error(train_all_y, Rtrain_all_pred))
        print("epoch " + str(epoch) + ": train MAE: " + str(train_mae) + '; RMSE: ' + str(train_rmse))
        #never-seen-experiment
        if cold_start:
            torch.save(model, 'model_save/model_save_'+train_dataset+'/never-seen/DSFlex/cnn_gcn_' + str(epoch) + '.pt')
        else:
            torch.save(model, 'model_save/model_save_'+train_dataset+'/all-data/DSFlex/cnn_gcn_' + str(epoch) + '.pt')

if __name__ == '__main__':
    BreakPoint=0
    #trainset={'SVM', 'PCA', 'SCC', 'LoR', 'DT', 'LP', 'PR', 'MF','LiR', 'TC', 'KM'}
    #validset={'SP', 'CC', 'SVD', 'PO'}

    trainset={'PCA','LoR','DT','LP','PR','MF','TC','KM','CC','PO'}
    validset={'SP','SCC','SVD','SVM','LiR'}

    #trainset={}
    #validset={}
    #train and validate
    #train_with_validate(BreakPoint,train_dataset='4',validate_dataset='4',cold_start=False,trainset=trainset,validset=validset)
    
    #validate without train
    #validate_without_train(train_dataset='2',validate_dataset='4',epoch_number=200,cold_start=False,trainset=trainset,validset=validset)

    train_without_validate(BreakPoint,train_dataset='4',validate_dataset='4',cold_start=False,trainset=trainset,validset=validset)




