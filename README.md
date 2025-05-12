**一、Data sample**
**(1) Extract data from the spark platform**
python sample/bo_sample.py   <path_to_spark_bench_folders>
For example: python3 sample/bo_sample.py   /usr/local/home/spark-bench/

bo_same.py runs 15 applications using BO ,including ['ConnectedComponent', 'DecisionTree', 'KMeans', 'LabelPropagation', 'LinearRegression','LogisticRegression', 'PageRank','PCA', 'PregelOperation', 'ShortestPaths', 'StronglyConnectedComponent', 'SVM', 'Terasort','TriangleCount']. Each application is run 500 times, among which 100 times are the initial random sampling and the remaining 400 times are the iterative search times.

**（2）The downloaded logs after the operation totaled 500*15= 7,500 pieces of data**

hdfs dfs -get <path_to_spark_history_server_folders_on_hdfs> <local_folders>
For example: hdfs dfs -get   hdfs://192.168.0.20:9000/spark/log  /usr/local/home/auto_tool/LITE-main/eventLogs/


**(3)process runing logs**
python scripts/**history_by_stage.py** <spark_bench_path> <history_dir> <result_path>

For example：
python3 sample/history_by_stage.py   /usr/local/home/spark-bench/ /usr/local/home/auto_tool/LITE-main/eventLogs/dataset_n/     /usr/local/home/auto_tool/LITE-main/dataset/MF_n.json n=1,2,3,4

**(4)Convert the raw data into training data**

python3 scripts/**build_dataset.py** <result_path> <dataset_path>  <dataset>

For example：
python3 ./**build_dataset.py**  /usr/local/home/auto_tool/LITE-main/ dataset/ /usr/local/home/auto_tool/LITE-main/dataset_n.csv   n 

n=1,2,3,4

Note:
The CSV file table only includes data features, environment features and Spark configuration parameter features.  
**(5)collect application feature**
**1.Build code dictionary**
python3 data_process_text.py
get  code.vocab
**2.Extract application features**
python3 dag2data.py
**3.Integrate the input features <d,e,p,a,sr> of the model together**
Specifically, d denotes the data features, e indicates the environment features, p refers to the Spark configuration parameter features, a represents the application features, and sr corresponds to runtimes of stages. 

python3 dataset_process.py


**二、Model train and validate**

**(1)Model train**
#config.py :   Define the hyperparameters of the model
#evaluation.py:  Evaluation model metrics include hr@5, ndcg@5 and MSE, etc.
#model.py ：Define the model structure

python mixed_train.py
train_without_validate(BreakPoint,train_dataset='4',validate_dataset='4',cold_start=False,trainset=trainset,validset=validset)
**note:**
"breakpoint" is used for breakpoint training
Train_dataset and validate_dataset are the data volume sizes of train_dataset and validate_dataset.
cold_start: Is it a cold start?
trainset is the application training set.
validset is the application validation set.


**(2)Model validate**

python model_pred.py

**(3)Model train and validate**
python mixed_train.py
train_without_validate(BreakPoint,train_dataset='4',validate_dataset='4',cold_start=False,trainset=trainset,validset=validset)



