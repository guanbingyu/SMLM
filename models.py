# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F



class SparkJobModel(nn.Module):
    def __init__(self, workload_feat_dim, stage_feat_dim,  hidden_dim, out_channels=64, kernel_sizes=[3, 4, 5]):
        super(SparkJobModel, self).__init__()
        self.workload_fc = nn.Linear(workload_feat_dim, hidden_dim)       
        self.final_fc= nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.InstanceNorm1d(hidden_dim,affine=False),
             nn.Linear(hidden_dim, hidden_dim),
            #nn.Linear(hidden_dim //2, hidden_dim // 4), 
            nn.Linear(hidden_dim, 1)
        )
        #V3
        self.stage_fc= nn.Sequential(
            nn.Linear(stage_feat_dim, hidden_dim), nn.InstanceNorm1d(hidden_dim,affine=False),
            nn.Linear(hidden_dim, hidden_dim)
        )
        #V4
        self.stage_leveL_fc = nn.Sequential(
            nn.Linear(stage_feat_dim, hidden_dim), 
            nn.Linear(hidden_dim, hidden_dim // 2), 
            nn.Linear(hidden_dim // 2, hidden_dim // 4), 
            nn.Linear(hidden_dim // 4, 1), 
        )

        #V2
        self.attention = nn.Sequential(
            nn.Linear(105, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 105),
            nn.Softmax(dim=1)
        )
        self.attention1 = nn.Sequential(
            nn.Linear(2*hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2*hidden_dim),
            nn.Softmax(dim=1)
        )

        self.adj_convs = nn.ModuleList([nn.Conv1d(64, out_channels, k) for k in kernel_sizes])
        self.node_convs = nn.ModuleList([nn.Conv1d(1, out_channels, k) for k in kernel_sizes])
        self.adj_fc = nn.Sequential(
            nn.Linear(192, hidden_dim), 
            nn.Linear(hidden_dim, hidden_dim // 2), 
            nn.Linear(hidden_dim // 2, hidden_dim // 4), 
        )
        #test 
        self.fc=nn.Linear(out_channels * len(kernel_sizes), out_channels)
    #The adjacency matrix features are extracted
    def compute_adj(self, adj):
        adj_out = [F.relu(conv(adj)) for conv in
                   self.adj_convs]  # len(kernel_sizes) * (batch_size, k_num, seq_len-k)  3*(64,64,(1000-k))
        adj_out = [F.max_pool1d(line, line.size(2)).squeeze(2) for line in
                   adj_out]  # len(kernel_sizes)*(batch_size, k_num)  3*(64,64)
        adj_out = torch.cat(adj_out,1)  # (batch_size, k_num*len(kernel_sizes))  (64, 64*3)
        adj_out = self.adj_fc(adj_out)
        return adj_out

    def compute_nodes(self, nodes):
        nodes = nodes.unsqueeze(1)  # [batch_size, 1, node_num]
        nodes_out = [F.relu(conv(nodes)) for conv in
                     self.node_convs]  # len(kernel_sizes) * (batch_size, k_num, seq_len-k)  3*(64,64,(1000-k))
        nodes_out = [F.max_pool1d(line, line.size(2)).squeeze(2) for line in
                     nodes_out]  # len(kernel_sizes)*(batch_size, k_num)  3*(64,64)
        nodes_out = torch.cat(nodes_out, 1)  # (batch_size, k_num*len(kernel_sizes))  (64, 64*3)
        nodes_out=self.fc(nodes_out)
        return nodes_out


    def forward(self, workload_feats, stage_feats, dag_nodes, dag_adjs,y):
        # 1. 处理外层特征
        workload_out = self.workload_fc(workload_feats.cuda())
        mse_loss=nn.MSELoss()
        combined_feats = []
        loss_mean=0
        for  nodes,adjs,y,workload_feat in zip( dag_nodes,dag_adjs,y,workload_feats):  # 遍历每条样本的stage特征和节点特征列表
            node_tensors = [torch.tensor(n, dtype=torch.float32).cuda() for n in nodes]
            adj_tensors = [torch.tensor(a, dtype=torch.float32).cuda() for a in adjs]
            y_tensors = [torch.tensor(a, dtype=torch.float32).cuda() for a in y]
            #node_tensors = [torch.tensor(n, dtype=torch.float32) for n in nodes]
            #adj_tensors = [torch.tensor(a, dtype=torch.float32) for a in adjs]
            #y_tensors = [torch.tensor(a, dtype=torch.float32) for a in y]
            workload_feat=workload_feat.unsqueeze(0).repeat(len(adjs),1).cuda()

            adj_tensors = torch.stack(adj_tensors).view(-1, 64, 64)
            adj_out = self.compute_adj(adj_tensors)#[8,16]
            #combined_tensors = self.compute_nodes(torch.stack(node_tensors))#[4,192] 横向拼接stage特征和节点特征，并转换为二维tensor
            combined_tensors=torch.stack(node_tensors)
            #combined_embeds = torch.stack([self.stage_fc(t) for t in combined_tensors])
            combined_tensors=torch.cat((combined_tensors,adj_out),dim=-1)
            #
            combined_tensors=torch.cat((combined_tensors,workload_feat),dim=-1)
            #加入attention机制
            weights = self.attention(combined_tensors)
            combined_tensors = combined_tensors * weights

            combined_embeds=self.stage_leveL_fc(combined_tensors)
            stage_runtimes = combined_embeds.transpose(0, 1)  # Flatten the tensor

            if stage_runtimes.size(1) < 64:
                padding = torch.zeros(1, 64 - stage_runtimes.size(1)).cuda()
                #padding = torch.zeros(1, 64 - stage_runtimes.size(1))
                stage_runtimes = torch.cat((stage_runtimes, padding), dim=1)
            else:
                stage_runtimes = stage_runtimes[:,:64]
            combined_feats.append(stage_runtimes)  
            loss = mse_loss(torch.stack(y_tensors).unsqueeze(1), combined_embeds)
            #print(loss.item())
            loss_mean+=loss.item()
            #通过一个全连接层，将stage和node信息来预测stage的运行时间
        combined_feat_out = torch.cat(combined_feats, dim=0)
        #对loss进行平均
        loss_mean=loss_mean/len(stage_feats)


        # 4. 合并特征
        #try:
           # combined = torch.cat((workload_out.cuda(), combined_feat_out), dim=-1)
        #except:
           # print('error')
        #加入attention机制
        #weights = self.attention1(combined)
        #combined = combined * weights
        #combined =  torch.matmul(weights.transpose(0,1),combined)
        #output = self.final_fc(combined)
        output = self.final_fc(combined_feat_out)
        return output,loss_mean


