# 基于gnn的模型

## 更新02说明
- 数据支持更新，目前可以支持v6.27及之后的数据版本的smiles数据，同时取消了对过时数据和FASTA版本的支持，如有需要直接跟我说，我再加上，使用方法见`训练和评估`部分。
- 增加了自动保存处理好的图文件，和自动读取指定图文件的功能，避免重复处理图文件带来的额外开销，使用方法见`训练和评估`部分。
- 增加了指定数据划分的功能，可以指定采用某个或批量运行某几个数据划分，使用方法见`训练和评估`部分。

## 更新01说明
- 更新了训练和评估命令，可以通过model_list来指定要运行的单个或多个模型，详情见`训练和评估`部分
- 支持SMILES格式的非天然氨基酸数据并且可以直接读取smiles格式数据，可以通过在程序运行命令添加`--is_smiles True`来直接读取smiles数据（默认为False，读取FASTA格式数据）

## 实验环境
基本conda环境配置与lm-based方法相同

额外环境
```
torch-geometric==2.6.1
torch_scatter==2.1.2+pt25cu121
ogb==1.3.6
```
## 训练和评估(命令有更新！！！)

```
# Run with default config.
# $DATASET, $GNN_TYPE, and $FILENAME are described below.
python main_pyg.py --activity $ACTIVITY --dataset $DATASET --model_list $['GNN_TYPE','GNN_TYPE',...] --splits $['SPLIT','SPLIT',...]
```

### `$ACTIVITY`
`$ACTIVITY`是指定一级活性，可选项为：
- `ADME`
- `Theraputic-AMP`
- `Theraputic-Other`
- `Tox`

### `$DATASET`
`$DATASET` 具体的数据集，例如：
- `ACE-APML`
- `Aox-APML`

### `$GNN_TYPE`
`$GNN_TYPE` specified the GNN architecture. It should be one of the followings:
- `gin`: GIN 
- `gcn`: GCN 
- `gat`: GAT
- `transformer`: Graph Transformer

### `$SPLIT`
`$SPLIT` specified the data split. 两部分组成，第一部分是划分方式，第二部分为seed，例如:
- `random1` :代表random划分，随机种子为seed_1
- `mmseqs22` :代表mmseqs2划分，随机种子为seed_2 

## 注意！！！
- 程序默认会根据指定的数据集把所有的模型跑一遍，并进行简单的调参，选出最优的结果。
- 因为初版程序为了方便快速调参，直接使用for循环对设置和参数进行了遍历。
- 如果希望指定参数，请修改main_pyg.py的146和147行。
- 后续会上述功能加入argument，以方便程序运行更个性化的指令。
- 由于整合工作量较大，后续pepland模型将单开一桌，嘻嘻。
- 后续将加入对FASTA格式的支持。
- 后续以需求判断是否加入meta_data方便读取指定数据与任务。
