# 基于gnn的模型
## 实验环境
基本conda环境配置与lm-based方法相同

额外环境
```
torch-geometric==2.6.1
torch_scatter==2.1.2+pt25cu121
ogb==1.3.6
```
## 训练和评估

```
# Run with default config.
# $DATASET, $GNN_TYPE, and $FILENAME are described below.
python main_pyg.py --activity $ACTIVITY --dataset $DATASET --gnn $GNN_TYPE
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
- 如果希望指定特定模型，请注释main_pyg.py的141行和145行，并调整代码缩进。
- 如果希望指定数据划分，请修改main_pyg.py的131行。
- 如果希望指定参数，请修改main_pyg.py的146和147行。
- 后续会上述功能加入argument，以方便程序运行更个性化的指令。