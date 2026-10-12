# Urban Planning Model

本项目实现了基于异构城市图与两级 Transformer 的统一城市决策模型，涵盖土地规划、道路规划、POI 选址和交通信号控制四类任务。它们的动作空间和优化目标不同，但都依赖同一套城市空间结构。我们将地块、道路和交叉口放进一张图里，再让共享策略网络在任务条件下逐步作出决策。

项目围绕这四类任务组织了从 OpenStreetMap 数据处理、专家轨迹生成到模型训练与仿真评估的流程，将静态空间规划与动态交通控制纳入同一套图表示和序列决策框架。

## 方法

模型分为实体编码和决策编码两层。下层 Transformer 编码地块、道路与交叉口的静态属性，以及信号任务中的动态交通观测；注意力通过图邻接关系限制在实体自身和一跳邻居之间。上层 Transformer 接收任务信息、全局表示和状态—动作历史，用因果注意力生成当前决策表示。信号任务还使用 GRU 压缩各路口的观测与动作历史。

土地与道路任务通过 pointer 选择图中的实体。土地规划按给定的用地类型顺序选择未分配地块，并检查剩余分配是否仍能满足面积配额；道路规划从完整候选路网开始逐步删除道路，用合法动作掩码维护覆盖、连通性和路口度约束。信号控制则为每个路口输出一个相位分布，在同一时刻执行所有路口的动作，并在 SUMO 中推进仿真。

训练先进行专家行为克隆，再使用 PPO 优化任务回报。静态规划的专家来自 Gurobi，信号控制的专家采用 MaxPressure。道路专家给出的是删除集合，而不是唯一的删除顺序，因此监督损失最大化当前合法专家集合上的总概率质量。PPO 阶段使用 GAE、任务内优势标准化和专家回报尺度归一化，并提供参考策略 KL 正则与更新早停配置。

土地回报衡量住宅周边的服务覆盖和绿地覆盖，道路回报由道路密度、连通性与间距指标组成，POI 选址在数量约束下优化地块面积与交通可达性构成的选址效用，信号回报使用路口平均停车车辆数的负值。具体定义见 [模型与训练](docs/模型与训练.md)，输入字段和轨迹格式见 [数据格式](docs/数据格式.md)。

## 环境准备

项目使用 Python 3.13 及以上版本，通过 uv 管理依赖。在项目根目录执行：

```bash
git clone https://github.com/iurww/urban-planning-model.git
cd urban-planning-model
uv sync

uv run up-model-data --help
uv run up-model-train --help
uv run up-model-eval --help
```

生成静态专家数据需要可用的 Gurobi license；交通数据采集与信号任务训练需要 SUMO。依赖中包含 `eclipse-sumo`、`sumolib` 和 `traci`。训练可以使用 CPU 或 CUDA，设备、并行仿真数量和日志设置都可以在配置文件中调整。

## 数据准备

数据处理从本地 `.osm.pbf` 文件开始。`batch-osm` 将城市切成 tile，重建道路、地块和交叉口，并生成用于 SUMO 的交通契约。随后通过 `prepare-training-data` 筛选拓扑、检查土地配额可行性，并按物理区域分组，避免重复地理范围进入不同数据池。

下面以天津的一组 4 km × 4 km tile 为例。PBF 路径、投影和网格范围需要根据自己的城市修改。

```bash
uv run up-model-data batch-osm \
  --pbf data/pbf/tianjin.osm.pbf \
  --city Tianjin --city-slug tianjin --crs EPSG:32650 \
  --origin-x 0 --origin-y 0 \
  --col-range 126 138 --row-range 1082 1094 \
  --size-m 4000 --cache-dir data/final --workers 4
```

候选 tile 下的 `evidence/` 保存 OSM 处理结果，`abstract/` 保存模型读取的拓扑及专家数据。筛选后，`data/sft/`、`data/ppo/` 和 `data/test/` 通过相对符号链接指向选中的 tile。静态专家由 `solve-all-real` 生成，交通需求和 MaxPressure 轨迹由 `sumo-collect-batch` 生成，`visualize-training-data` 可以查看拓扑和专家方案。各阶段的完整命令见 [使用指南](docs/使用指南.md)。

道路和交通环境是基于 OSM 重建的抽象路网：车道按每方向两到三条归一化，交叉口使用统一的相位定义，交通需求由程序生成。实验中的排队和等待数据来自 SUMO 仿真，不能直接理解为真实城市交通测量。

## 训练与评估

准备好拓扑和专家轨迹后，先运行 SFT，再用得到的 checkpoint 初始化 PPO。默认模型为隐藏维度 128、两层实体编码器和两层决策编码器，每层四个注意力头。完整参数位于 `src/configs/train.json`。

```bash
uv run up-model-train \
  --config src/configs/train.json --stage sft \
  --data-root data/sft --test-data-root data/test \
  --output-dir outputs/sft --wandb-mode disabled \
  --fail-if-output-exists

uv run up-model-train \
  --config src/configs/train.json --stage ppo \
  --data-root data/ppo --test-data-root data/test \
  --init-checkpoint outputs/sft/sft_best.pt \
  --output-dir outputs/ppo --wandb-mode disabled \
  --fail-if-output-exists
```

如果只是检查本地训练流程，可以将配置换成 `src/configs/smoke_train.json`。它缩小了模型和训练规模，但仍需要准备数据、专家轨迹及相应的运行依赖。

训练会记录解析后的配置、数据划分、输入文件快照和 checkpoint。独立评估入口可以加载 checkpoint，输出逐任务、逐 tile 的回报及与专家的对比：

```bash
uv run up-model-eval \
  --checkpoint outputs/ppo/ppo_best.pt \
  --data-root data/test --tasks land road signal \
  --episodes 1 --seed 0 --device auto \
  --output outputs/ppo/evaluation.json \
  --fail-if-output-exists --wandb-mode disabled
```

当前实现会在训练期间使用 `--test-data-root` 指定的数据选择最佳 checkpoint，没有另设内部 validation 划分。因此，这个数据池实际承担模型选择的角色；如果要报告独立泛化结果，应另外保留一个未参与训练和 checkpoint 选择的评估池，再通过 `up-model-eval --data-root` 指向它。`--tile-ids` 模式会复用指定拓扑进行训练和测试，适合收敛诊断。

## 代码组织

`scripts/` 负责 OSM 拓扑处理、Gurobi 求解、SUMO 数据采集和数据可视化；`src/upmodel/` 包含图数据加载、策略网络、规划环境、SFT/PPO 训练器和评估逻辑。`tests/` 覆盖拓扑与数据划分、批处理策略、信号相位切换、训练初始化和评估产物等行为，可以运行：

```bash
uv run python -m unittest discover -s tests
```

仓库提供代码和配置，原始 PBF、生成的数据集、训练权重和实验输出需要自行准备。阅读方法实现可以从 `src/upmodel/model/` 开始；复现数据到训练的流程则建议先看使用指南。当前版本着重于共享策略的实现与训练流程，模型性能、跨城市迁移和多任务收益仍需要具体实验验证。
