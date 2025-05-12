# -*- coding: utf-8 -*-
import os
import json
import torch
import numpy as np
from torch_geometric.data import Data
from argparse import ArgumentParser

#def parse_args():
    #parser = ArgumentParser(description="get graph of dag")

   # parser.add_argument('dag_logs', type=str, help='dir of example log of each workload')

    #return parser.parse_args()

#args = parse_args()

#dag_logs = args.dag_logs

spark_conf_names = ['spark.default.parallelism', 'spark.driver.cores', 'spark.driver.memory',
                    'spark.driver.maxResultSize',
                    'spark.executor.instances', 'spark.executor.cores', 'spark.executor.memory',
                    'spark.executor.memoryOverhead',
                    'spark.files.maxPartitionBytes', 'spark.memory.fraction', 'spark.memory.storageFraction',
                    'spark.reducer.maxSizeInFlight',
                    'spark.shuffle.compress', 'spark.shuffle.file.buffer', 'spark.shuffle.spill.compress']


def read_dag(dags):
    dags = sorted(dags.items(), key=lambda x: int(x[0].split('_')[2]))
    processed_dags = {}
    name_all = []
    for stage_id, dag in dags:
        processed_dags[stage_id] = []
        for node in dag:
            #print(node)
            p_node = {}
            p_node['id'] = node['rdd_id']
            p_node['parent_ids'] = node['parent_ids']

            # Check if 'scope' key exists
            scope = node.get('scope')
            if scope is None or not isinstance(scope, dict) or 'name' not in scope:
                p_node['name'] = ''
            else:
                p_node['name'] = scope['name']
                name_all.append(scope['name'])

            processed_dags[stage_id].append(p_node)
    return processed_dags



def update_child(dags):
    for stage_id, dag in dags.items():
        for node_id, node in dag.items():
            if len(node['parent_ids']) > 0:
                for parent_id in node['parent_ids']:
                    if parent_id not in dag.keys():
                        node['parent_ids'] = []
                        continue
                    if 'child_ids' not in dag[parent_id].keys():
                        dag[parent_id]['child_ids'] = []
                    dag[parent_id]['child_ids'].append(node_id)


def traverse_all_dag(all_dags, word_count):
    dags = read_dag(all_dags)  # read_dag 返回一个包含列表的字典
    for stage_id, dag in dags.items():  # 遍历字典
        for node in dag:  # 遍历列表中的节点
            if node['name'] != '':
                word_count[node['name']] = word_count.get(node['name'], 0) + 1



def build_vocab():
    raw_file_list = os.listdir(dag_logs)
    word_count = {}
    for f in raw_file_list:
        raw_file = open(dag_logs + f, 'r', encoding='utf-8')
        line = raw_file.readlines()[0]
        line_json = json.loads(line)
        all_dags = line_json['dags']
        traverse_all_dag(all_dags, word_count)
    word_count = list(word_count.items())
    word_count.sort(key=lambda k: k[1], reverse=True)
    write = open('dag_data/vocab', 'w', encoding='utf-8')
    for word_pair in word_count:
        write.write(word_pair[0] + '\t' + str(word_pair[1]) + '\n')
    write.close()


def get_w2i():
    vocab_file = open('dag_data/vocab', encoding='utf-8')
    w2i = {}
    i = 2
    for pair in vocab_file:
        v = pair.split('\t')[0]
        w2i[v] = i
        i += 1
    return w2i


def dag2data(dag, w2i):
    x = []
    src_idx = []
    tgt_idx = []
    node_id2index = {}
    for idx, node in enumerate(dag):
        node_id2index[node['id']] = idx
    for idx, node in enumerate(dag):
        if node['name'] not in w2i.keys():
            continue
        x.append(w2i[node['name']])
        if len(node['parent_ids']) == 0:
            continue
        else:
            for parent_id in node['parent_ids']:
                if parent_id not in node_id2index.keys():
                    continue
                src_idx.append(node_id2index[parent_id])
                tgt_idx.append(idx)
    print(len(x))
    x = torch.LongTensor(x).unsqueeze(1)
    edge_index = torch.tensor([src_idx, tgt_idx], dtype=torch.long)
    y = torch.FloatTensor(0)
    data = Data(x=x, edge_index=edge_index, y=y)
    return data


def dag2matrix(dag, w2i):
    nodes = np.zeros(64, dtype=float)
    node_id2index = {}
    for idx, node in enumerate(dag):
        node_id2index[node['id']] = idx
    #print()
    adj = np.zeros([64, 64], dtype=float)
    for idx, node in enumerate(dag):
        if node['name'] not in w2i.keys() or idx >= 64:
            continue
        nodes[idx] = w2i[node['name']]
        if len(node['parent_ids']) == 0:
            continue
        else:
            for parent_id in node['parent_ids']:
                if parent_id not in node_id2index.keys():
                    continue
                if idx >= 64 or node_id2index[parent_id] >= 64:
                    continue
                adj[node_id2index[parent_id], idx] = 1
    return nodes, adj

#2024.12.2添加stage(RDD)之间关系提取函数。
def dag2matrix_RDD(dag, w2i,s_id):
    nodes = np.zeros(64, dtype=float)
    node_id2index = {}
    node_index = {}
    for idx, node in enumerate(dag):
        node_id2index[node['id']] = idx
        node_index[node['id']] = s_id
    adj = np.zeros([64, 64], dtype=float)
    for idx, node in enumerate(dag):
        if node['id']<64:
            adj[node['id'], node['id']] = 1
        if node['name'] not in w2i.keys() or idx >= 64:
            continue
        nodes[idx] = w2i[node['name']]
        if len(node['parent_ids']) == 0:
            continue
        else:
            for parent_id in node['parent_ids']:
                #去除了parent_id和RDD不在64个RDD中
                if node['id']>=64 or parent_id >= 64:
                    continue
                adj[node['id'], parent_id] = 1
    return nodes, adj, node_index


#2024.12.2 通过索引获取stage(RDD)之间的关系
def Dag2Matrix_Stage(s_id,dag,ws2node_index):
    s_id=int(s_id)
    i=0
    adj = np.zeros([64, 64], dtype=float)
    if s_id < 64:
        adj[s_id, s_id] = 1
    for idx, node in enumerate(dag):        
        for parent_id in node['parent_ids']:
            if int(ws2node_index[parent_id])>=64:
                continue
            else:
                print(ws2node_index[parent_id])
                adj[s_id, int(ws2node_index[parent_id])] = 1
                i=i+1
    return adj


def build_graph_data():
    raw_file_list = os.listdir(dag_logs)
    ws2g = {}
    w2i = get_w2i()
    for f in raw_file_list:
        raw_file = open(dag_logs + f, 'r', encoding='utf-8')
        line = raw_file.readlines()[0]
        line_json = json.loads(line)
        all_dags = line_json['dags']
        all_dags = read_dag(all_dags)
        ws2g[f] = {}
        for stage_id, dag in all_dags.items():
            graph_data = dag2data(dag, w2i)
            s_id = stage_id.split('_')[0]
            ws2g[f][s_id] = graph_data
    torch.save(ws2g, 'dag_data/ws2g.pth')


def gen_data(history_dir, dataset_path):
    count = 0

    #os.chdir(history_dir)
    his_file_list = os.listdir(history_dir)
    # 对历史记录按文件名排序
    his_file_list.sort()
    print(len(his_file_list))
    # 逐条进行读取
    for path in his_file_list:
        print(path)
        dataset_file = open(dataset_path+path+'.json', 'w', encoding='utf-8')
        if path.endswith('inprogress'):
            continue
        his_file = open(history_dir+path, encoding='utf-8')
        # 处理一条历史记录
        one_data = {}
        dags = {}
        all_stage_info = {}
        cur_metrics, cur_stage = {'input': 0, 'output': 0, 'read': 0, 'write': 0}, 0
        start_timestamp = None
        end_timestamp = None
        transformed_data = {}
        for line in his_file:
            try:
                line_json = json.loads(line)
            except:
                print('json错误')
                continue
            # 统计每个stage的shuffle read/write、input/output，需要每个task累加得到stage
            if line_json['Event'] == 'SparkListenerTaskEnd':
                cur_stage = line_json['Stage ID']
                # 新的stage
                if line_json['Stage ID'] not in all_stage_info:
                    all_stage_info[cur_stage] = {'input': 0, 'output': 0, 'read': 0, 'write': 0}
                # if line_json['Stage ID'] != cur_stage:
                #     cur_metrics, cur_stage = {'input': 0, 'output': 0, 'read': 0, 'write': 0}, line_json['Stage ID']
                try:
                    all_stage_info[cur_stage]['input'] += line_json['Task Metrics']['Input Metrics']['Bytes Read']
                    all_stage_info[cur_stage]['output'] += line_json['Task Metrics']['Output Metrics']['Bytes Written']
                    all_stage_info[cur_stage]['read'] += (line_json['Task Metrics']['Shuffle Read Metrics']['Remote Bytes Read'] +
                                            line_json['Task Metrics']['Shuffle Read Metrics']['Local Bytes Read'])
                    all_stage_info[cur_stage]['write'] += line_json['Task Metrics']['Shuffle Write Metrics']['Shuffle Bytes Written']
                except:
                    print('metrics key error')
                    #continue,直接去掉这种
                    break

            if line_json['Event'] == 'SparkListenerEnvironmentUpdate':
                spark_props = line_json['Spark Properties']
                conf_list = []
                for conf_name in spark_conf_names:
                    conf_list.append(conf_name + '=' + spark_props[conf_name])
                one_data['SparkParameters'] = conf_list
            if line_json['Event'] == 'SparkListenerApplicationStart':
                start_timestamp = line_json['Timestamp']
                one_data['AppName'] = line_json['App Name']
                one_data['AppId'] = line_json['App ID']
            if line_json['Event'] == 'SparkListenerApplicationEnd':
                end_timestamp = line_json['Timestamp']


            #获取日志中的dags信息
            if line_json['Event'] == 'SparkListenerJobStart':

                for stage_info in line_json.get("Stage Infos", []):
                    stage_id = stage_info["Stage ID"]
                    stage_key = f"dag_stage_{stage_id}"

                    transformed_data[stage_key] = []

                    for rdd_info in stage_info.get("RDD Info", []):
                        #print(rdd_info)
                        # 检查每个字段是否存在
                        rdd_entry = {}
                        if "Callsite" in rdd_info:
                            rdd_entry["call_site"] = rdd_info["Callsite"]
                        if "Name" in rdd_info:
                            rdd_entry["name"] = rdd_info["Name"]
                        if "Parent IDs" in rdd_info:
                            rdd_entry["parent_ids"] = rdd_info["Parent IDs"]
                        if "RDD ID" in rdd_info:
                            rdd_entry["rdd_id"] = rdd_info["RDD ID"]
                        if "Scope" in rdd_info:
                            try:
                                rdd_entry["scope"] = json.loads(rdd_info["Scope"])  # 转换为 JSON 对象
                            except json.JSONDecodeError:
                                pass  # 如果 Scope 解析失败，跳过
                        # 如果 rdd_entry 不为空，才添加到结果中
                        if rdd_entry:
                            #print(rdd_entry)
                            #print(transformed_data)
                            transformed_data[stage_key].append(rdd_entry)
                            #print(transformed_data)



            # 按stage获取执行时间，shuffle read/write、input/output
            if line_json['Event'] == 'SparkListenerStageCompleted':
                stage_id = 'dag_stage_' + str(line_json['Stage Info']['Stage ID'])
                try:
                    stage_id = line_json['Stage Info']['Stage ID']
                    all_stage_info[stage_id] = cur_metrics
                    stage_start = line_json['Stage Info']['Submission Time']
                    stage_end = line_json['Stage Info']['Completion Time']
                    all_stage_info[stage_id]['duration'] = int(stage_end) - int(stage_start)
                except:
                    print('stage duration key error')
                    continue

        one_data['StageInfo'] = all_stage_info
        #print(transformed_data)
        one_data['dags']=transformed_data
        dataset_file.write(json.dumps(one_data, sort_keys=True) + '\n')
        count += 1
    print('samples count: ' + str(count))

#2024.12.2添加stage之间关系提取函数。
def build_stage_matrix(dag_logs):
    raw_file_list = os.listdir(dag_logs)
    ws2nodes, ws2adj,ws2node_index,wstage_adj = {},{},{},{}
    w2i = get_w2i()
    for f in raw_file_list:
        raw_file = open(dag_logs + f, 'r', encoding='utf-8')
        line = raw_file.readlines()[0]
        line_json = json.loads(line)
        #dags=read_dags(line_json)
        all_dags = line_json['dags']
        all_dags = read_dag(all_dags)
        ws2nodes[f], ws2adj[f],ws2node_index[f] = {}, {},{}
        for stage_id, dag in all_dags.items():
            s_id = stage_id.split('_')[-1]
            nodes, adj, node_index = dag2matrix_RDD(dag, w2i,s_id)
            #只提取内部RDD之间的关系
            #nodes, adj = dag2matrix(dag, w2i)
            
            ws2nodes[f][s_id] = nodes
            ws2adj[f][s_id] = adj
            ws2node_index[f] ={**ws2node_index[f],**node_index}
        wstage_adj[f] = {}
        for stage_id, dag in all_dags.items():
            s_id = stage_id.split('_')[-1]
            stage_adj=Dag2Matrix_Stage(s_id,dag,ws2node_index[f])
            wstage_adj[f][s_id] = stage_adj
  
    torch.save(ws2nodes, 'dag_data/ws2nodes.pth')
    torch.save(wstage_adj, 'dag_data/ws2adj.pth')

if __name__ == '__main__':
    dag_logs='dag_logs/'
    #用来处理原始数据
    #gen_data(dag_logs,'dag_logs_raw/')
    #统计dag中name中各个关键词出现的频次
    #build_vocab()
    #build_graph_data()
    #2024.12.2添加stage之间关系提取函数。
    build_stage_matrix(dag_logs)
