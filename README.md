# apple_detection

苹果可见区域／完整果实双分支实例分割研究。当前开发以 Mask2Former 为基础，服务于矮化密植果园中的识别与后续定位。

## 验收与已测基线

目标是在同一置信阈值下 Precision、Recall 均达到 95%，并争取 Precision 接近 98%。目前没有达到这个目标。

已完成的查询式双分支基线在完整 PApple VAL（814 图、3449 GT）上，BOX Precision/Recall 均为 88.3444%。随后增加几何修正、辅助监督及分辨率的联合实验，截至第 3 轮最高 P/R 为 87.5906%。这属于开发集结果，不是独立部署表现；部分标注异常和重遮挡目标均保留。

数值记录见 `benchmarks/baseline_metrics.json`，历史执行源码见 `reference/local_amodal/`。参考快照按原字节保存，供追溯，不是迁移后的运行入口；它引用的旧本地数据、权重、监督报告没有上传。

## 新开发路线

新代码位于 `src/apple_seg/`，与仍在运行的历史实验隔离：

1. 从已经完成的查询式双分支 checkpoint 严格加载模型参数，使用原 768 输入尺度作为起点。
2. PApple 提供真实 V/A 配对监督；MinneApple 提供 V-only 监督。未知 A 不参与匹配、损失、负样本或伪补全监督。
3. 对密集场景采用预先固定的重叠切片，保留全部原 GT；整图评估需恢复原坐标并合并重复实例。
4. 使用完整验证集比较，固定模型和阈值后再做外部验证。CPU 契约测试、GPU 可运行性、完整 P/R 分开记录。

这条路线尚在实现和验证，不能将源码存在或测试通过当成精度提升。

## 环境与数据

当前实测环境为 Windows、Python 3.11.15、PyTorch 2.12.1+cu132、Transformers 4.57.1。依赖列在 `requirements.txt`，实测 CUDA 构建列在 `constraints-tested.txt`。PyTorch/torchvision 的 CUDA 构建需要与机器驱动匹配，参考 [PyTorch 官方安装页](https://pytorch.org/get-started/locally/)。新实现沿用现有 F 盘环境进行测试。

数据和大文件不进入 Git。公开来源为 [PApple v4](https://zenodo.org/records/7260694) 和 [MinneApple](https://github.com/nicolaihaeni/MinneApple)。仓库只记录来源、配置和哈希；本机路径写入被忽略的 `configs/local.json`。官方测试数据不参与训练或阈值选择。

`reference/` 是历史快照，`src/` 是新开发入口。所有权重、数据集、环境、缓存及完整运行输出保留在 F 盘，不通过普通 Git 上传。
