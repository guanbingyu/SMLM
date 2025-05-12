import torch
import numpy as np
from torch.utils.data import Dataset


class TransDataset(Dataset):
    def __init__(self, feat, code, y, dag_nodes, dag_adjs, label):
        self.feat = feat
        self.code = code
        self.y = y
        self.dag_nodes = dag_nodes
        self.dag_adjs = dag_adjs
        self.label = label

    def __getitem__(self, index):
        feat, code, y = self.feat[index], self.code[index], self.y[index]
        dag_node, dag_adj = self.dag_nodes[index], self.dag_adjs[index]
        label = self.label[index]
        return feat, code, y, dag_node, dag_adj, label

    def __len__(self):
        return len(self.y)


class CodeDagDataset(Dataset):
    def __init__(self, feat, code, y, dag_nodes, dag_adjs,W):
        self.feat = feat
        self.code = code
        self.y = y
        self.dag_nodes = dag_nodes
        self.dag_adjs = dag_adjs
        self.W = W

    def __getitem__(self, index):
        feat, code, y = self.feat[index], self.code[index], self.y[index]
        dag_node, dag_adj = self.dag_nodes[index], self.dag_adjs[index]
        w = self.W[index]
        return feat, code, y, dag_node, dag_adj, w

    def __len__(self):
        return len(self.y)

class DagDataset(Dataset):
    def __init__(self, feat,  y, dag_nodes, dag_adjs):
        self.feat = feat
        self.y = y
        self.dag_nodes = dag_nodes
        self.dag_adjs = dag_adjs

    def __getitem__(self, index):
        feat,  y = self.feat[index],  self.y[index]
        dag_node, dag_adj = self.dag_nodes[index], self.dag_adjs[index]
        return feat,  y, dag_node, dag_adj

    def __len__(self):
        return len(self.y)

class NodeData(Dataset):
    def __init__(self, feat,  y, dag_nodes):
        self.feat = feat
        self.y = y
        self.dag_nodes = dag_nodes

    def __getitem__(self, index):
        feat,  y = self.feat[index],  self.y[index]
        dag_node = self.dag_nodes[index]
        return feat,  y, dag_node

    def __len__(self):
        return len(self.y)

#2024-12-3 构建混合粒度的数据集
class SparkJobDataset(Dataset):
    def __init__(self, workload_feat, labels, stage_feats, dag_nodes, dag_adjs,ys,W):
        """
        workload_feat: 外层特征 (numpy array)
        labels: 标签 (numpy array)
        stage_feats: 每条外层特征对应多个stage特征 (list of lists of 1D numpy arrays)
        dag_nodes: 每条外层特征对应多个节点特征 (list of lists of 1D numpy arrays)
        dag_adjs: 每条外层特征对应多个邻接矩阵 (list of lists of 2D numpy arrays)
        """
        assert len(workload_feat) == len(labels) == len(stage_feats) == len(dag_nodes) == len(dag_adjs)==len(ys)==len(W), \
            "外层特征数量不一致"
        
        self.workload_feat = torch.tensor(workload_feat, dtype=torch.float32)
        #self.workload_feat = workload_feat
        self.labels = torch.tensor(labels, dtype=torch.float32)
        self.stage_feats = stage_feats
        self.dag_nodes = dag_nodes
        self.dag_adjs = dag_adjs
        self.ys = ys
        self.W = W

    def __getitem__(self, index):
        workload_feat = self.workload_feat[index]
        label = self.labels[index]
        stage_feat = self.stage_feats[index]  # list of 1D numpy arrays
        dag_node = self.dag_nodes[index]  # list of 1D numpy arrays
        dag_adj = self.dag_adjs[index]  # list of 2D numpy arrays
        y = self.ys[index]
        w=self.W[index]
        return workload_feat, label, stage_feat, dag_node, dag_adj, y,w

    def __len__(self):
        return len(self.labels)

# Customize the collate_fn  
def collate_fn(batch):
    """
    batch: list of tuples (workload_feat, label, stage_feats, dag_nodes, dag_adjs)
    """
    workload_feats = torch.stack([item[0] for item in batch])
    #workload_feats = [item[0] for item in batch]
    labels = torch.stack([item[1] for item in batch])
    stage_feats = [item[2] for item in batch]
    dag_nodes = [item[3] for item in batch]
    dag_adjs = [item[4] for item in batch]
    ys = [item[5] for item in batch]
    W=[item[6] for item in batch]
    return workload_feats, labels, stage_feats, dag_nodes, dag_adjs, ys,W


class MyDataset(Dataset):
    def __init__(self, x, code, y):
        self.x = x
        self.code = code
        self.y = y

    def __getitem__(self, index):
        x, code, y = self.x[index], self.code[index], self.y[index]
        return x, code, y

    def __len__(self):
        return len(self.y)

