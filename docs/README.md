# UP-Model 文档

OSM 数据合成 Gurobi/SUMO 专家数据 训练数据格式 SFT + PPO 两阶段训练。



## 文档清单

1. [使用指南](./使用指南.md)：环境准备、逐阶段命令、全部命令行参数、输入输出路径。
2. [数据格式](./数据格式.md)：一个 tile 内每个文件的用途、JSON/CSV/XML 字段和跨文件 ID 约束。
3. [模型与训练](./模型与训练.md)：四类任务的数学定义、图与特征、两级 Transformer、SFT/PPO 公式。

正式训练参数位于 `src/configs/train.json`



## 当前能力边界

| 任务 | 专家数据生成 | SFT | PPO | 当前决策方式 |
|---|---:|---:|---:|---|
| 土地规划 | Gurobi | 是 | 是 | 给定用地类型（what），模型指向一个未分配地块（where） |
| 道路规划 | Gurobi | 是 | 是 | 给定“升级”，模型逐步指向需要升级的道路 |
| POI 选址 | Gurobi |  |      |                                                       |
| 红绿灯调度 | SUMO MaxPressure | 是 | 是 | 每 15 秒为全部信号路口联合选择 canonical phase |



## 标准数据流

```text
OSM .pbf
  -> batch-osm
  -> <dataset-root>/<candidate-dir>/<tile>/{evidence,abstract}
  -> prepare-training-data
  -> <dataset-root>/topology_selection.csv
  -> <dataset-root>/train/<tile> 与 <dataset-root>/test/<tile> 相对符号链接
  -> visualize-training-data（扫描全体候选；求解前仅有拓扑面板）
  -> solve-all-real (Land/Road/POI)
  -> sumo-collect-batch (OD + MaxPressure)
  -> visualize-training-data（刷新已有专家面板）
  -> up-model-train --stage sft
  -> up-model-train --stage ppo --init-checkpoint ...
  -> up-model-eval
```

`<dataset-root>` 是一套数据集的根目录，可以是正式数据的 `data/`，也可以是验证数据的
`datadoc/`。候选目录、`topology_selection.csv`、`train/` 和 `test/` 必须在同一个
数据集根内。正式训练目录中的 `data/train/<tile>` 和 `data/test/<tile>` 都是符号链接，
直接指向 `data/final/<tile>/abstract/`；`evidence/` 不进入模型。`data/train` 是候选训练池，
程序会划出内部 training 与 validation；`data/test` 只用于最终测试，不参与
checkpoint 选择。


