# 基于gnn的模型

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
python main_pyg.py --activity $ACTIVITY --dataset $DATASET --model_list $[GNN_TYPE,GNN_TYPE,...]
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

## 注意！！！
- 程序默认会根据指定的数据集把所有的模型跑一遍，并进行简单的调参，选出最优的结果。
- 因为初版程序为了方便快速调参，直接使用for循环对设置和参数进行了遍历。
- 如果希望指定数据划分，请修改main_pyg.py的131行。
- 如果希望指定参数，请修改main_pyg.py的146和147行。
- 后续会上述功能加入argument，以方便程序运行更个性化的指令。
- 后续整合pepland模型。
- 等待数据格式统一后，会调整smiles格式读取的代码，当前代码可能需要修改列名才能读取一些文件。
