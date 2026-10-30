# 原强模型运行源码副本

本目录用于复现交付的 ConvNeXt-Base / ConvNeXt-Small 检测与 V/H/A 分割模型，以及从原 Base 热启动的 C0/C1 实验。

来源是本机成果包 `含权重/苹果检测成果包_20261029/07_运行环境与源码`；原始成果包和两份权重不在本仓库修改。不能将这份副本称为旧训练提交的完整恢复：旧 `train_od.py` 和完整 Git 历史没有交付，原始 EMA shadow 也未完整保留。本轮记录新的源码 SHA 与实验来源。

唯一架构构造补丁位于 `src/apple_seg/od_model.py`：构造 ConvNeXt 时使用 `weights=None`，避免下载与当前模型无关的初始化权重；随后严格恢复交付 checkpoint 的全部参数并检查一致性。模型结构和现有头部不变。

版本管理包含全部 `src/apple_seg/*.py`、原 `tmp/flip_tta.py` 和 `reference/local_amodal/scripts/metrics.py`。本地复制的其他历史参考脚本不进入本轮训练或评估入口，也不提交。数据、权重、实验输出与缓存均不提交 Git。

完整复现期间不要编辑本目录内的 Python 源码或 `scripts/reproduce_strong_baseline.py`：运行前后会核验这些文件的 SHA，变化会使该次结果失效。
