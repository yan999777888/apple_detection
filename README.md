# apple_detection

面向矮化密植果园的苹果分割与视觉定位研究。当前采用两种职责不同的模型：原 ConvNeXt-Base＋Small 用于全局关键帧识别；局部 YOLO26s-seg 用于已锁定目标接近阶段的实时可见区域分割。原全局强模型保持保底，局部模型单独训练、评价与版本管理。

## 当前模型与证据

| 模块 | 当前方案 | 已验证的范围 |
|---|---|---|
| 全局识别 | 原 Base＋Small，原图/水平翻转四视图，V/H/A 掩膜模型 | 原 PApple VAL 框评价 P=94.4343%、R=90.0261%（TP3105/FP183/FN344）；不是像素分割 AP 或三维定位精度 |
| 局部分割 | YOLO26s-seg，COCO apple47 精确迁移为单类 apple0，人工 V 位图监督 | 12 步 GPU 工程压力通过；正式精度按新实验报告验收，不把烟测当精度提升 |
| 三维定位 | 深度稳健拟合及候选轮廓联合优化对照 | 合成诊断和小样本果径分析已记录；D405 真实果心误差与完整闭环尚未验证 |

局部目标为已锁定苹果每帧重新分割与定位达到 20 FPS。历史 COCO 候选的小样本分割速度不等于完整系统的 20 FPS；正式评价需报告遮挡召回、掩膜质量、P95 延迟和输入范围。

## 当前代码入口

- `strong_runtime/`：已交付原强模型运行源码、严格加载和原翻转推理；边界见其 README。旧 `train_od.py` 和完整旧训练历史没有交付，不能宣称已经恢复。
- `scripts/reproduce_strong_baseline.py`、`configs/strong_baseline_4060.json`：原强基线复现入口。
- `src/apple_strong/` 及强模型相关 `scripts/`：原强模型 C0/C1、部署提速和候选筛选实验代码。
- `src/apple_local_rgbd/`：局部分割数据适配、训练/评价及几何诊断；`scripts/runtime_local_rgbd.ps1` 是 F 盘环境入口。
- `scripts/prepare_local_apple_data.py`：按原 apple_ID 栅格化 V/A，原尺寸位图无损保存，不拆分同果多个可见片段。
- `scripts/run_local_apple_smoke.ps1`：有界工程压力入口，最多 32 步，不能作为长训入口。
- `scripts/train_local_segmenter_formal.py`、`scripts/evaluate_local_segmenter.py`：正式局部模型训练和评价入口；具体参数以 `--help` 和正式实验配置为准。
- `reference/strong_delivery_20261029/`：原交付历史文本源码与配置的原字节归档，附来源和 SHA，不作为默认入口。

代码对应关系见 [版本索引](docs/代码与实验版本索引_20261104.md)。原查询式、混合数据、切片路线和旧 `.cmd` 入口保留为历史实验，不能当作当前默认训练路线。

## 局部训练与评价约定

PApple train 为 2304 图/8579 实例/14 空图，val 为 814 图/3449 实例/13 空图。官方 test 807 图封存。人工可见掩膜 V 用于本轮分割监督，A 仅用于遮挡分层，不将教师估计的 A 冒充真值。

按 V/A 面积比划分重遮挡（<0.40）、中遮挡（0.40–0.70）、轻遮挡（0.70–0.90）、基本可见（0.90–1.00）；V 面积超过 A 的实例单列保留。train/val 有 309 个相同果实身份，因此 val 是开发验证集，不能称新果实独立测试。

首轮正式实验固定 10 个 epoch，在训练前和第 1、5、10 个 epoch 做完整验证。报告可见 mask 的 PR/AP50、固定阈值 P/R、逐 GT 遮挡漏检和匹配 mask IoU；不与原强模型的完整果实框指标直接混比。速度使用固定图片、FP16、预热和重复轮次、原尺寸 V 掩膜交付到 CPU 的协议，另报 P95，不包含相机采集或三维优化。

工程实测见 [局部训练准备与压力测试](docs/局部苹果模型训练准备与压力测试_20261104.md)。运行报告以各阶段落盘 JSON 为准，代码存在不代表某阶段已执行成功。

## 当前环境与 Git 边界

当前为 Windows、RTX 4060 Laptop、Python 3.11.15、PyTorch 2.12.1+cu132、Ultralytics 8.4.115。局部入口使用 F 盘 `envs/local_rgbd_v1`，只读复用 F 盘 falldet 的已安装依赖，版本快照为 `requirements-local-rgbd.txt`。不要向原强模型的共享环境盲目升级依赖。

代码、配置、依赖快照、方法与精选实测报告进入 Git。权重、RGB/深度数据、环境、缓存、完整预测输出和私有对话不通过普通 Git 上传；仓库记录来源与哈希，克隆 Git 不包含这些资产。当前新入口要求文件、环境和缓存均位于 F 盘。

数据来源：[PApple v4](https://zenodo.org/records/7260694)、历史使用的 [MinneApple](https://github.com/nicolaihaeni/MinneApple)。原资产和第三方许可按其来源保留。

<details>
<summary>早期查询式／切片路线的历史说明（非当前默认入口）</summary>

以下保留早期 README 的方法和复现说明，仅用于历史追溯。

## 历史验收与已测基线

目标是在同一置信阈值下 Precision、Recall 均达到 95%，并争取 Precision 接近 98%。目前没有达到这个目标。

已完成的查询式双分支基线在完整 PApple VAL（814 图、3449 GT）上，BOX Precision/Recall 均为 88.3444%。随后增加几何修正、辅助监督及分辨率的联合实验，截至第 3 轮最高 P/R 为 87.5906%。这属于开发集结果，不是独立部署表现；部分标注异常和重遮挡目标均保留。

数值记录见 `benchmarks/baseline_metrics.json`，历史执行源码见 `reference/local_amodal/`。参考快照按原字节保存，供追溯，不是迁移后的运行入口；它引用的旧本地数据、权重、监督报告没有上传。

## 新开发路线

新代码位于 `src/apple_seg/`，与历史实验隔离：

1. 从已经完成的查询式双分支 checkpoint 严格加载模型参数，使用原 768 输入尺度作为起点。
2. PApple 提供真实 V/A 配对监督；MinneApple 提供 V-only 监督。未知 A 不参与匹配、损失、负样本或伪补全监督。
3. MinneApple 每图统一使用六个 512×512 重叠切片，x 起点 0/208、y 起点 0/384/768。670 图真实容量检查最大 64 个实例，全部原 GT 至少有一个完整包含的切片，像素和身份不丢失。切片重复曝光不算新增原始 GT。
4. 每个成功更新混合 2 张 PApple 和 2 个 MinneApple 切片，默认 600 次成功更新；workers=0、逐图读取、CPU 原始掩膜分块采样。匹配及损失使用 FP32。
5. 每 200 次成功更新评估完整 PApple VAL，使用一个全局 BOX 阈值；另报同阈值的独立 V/A 掩膜 P/R。固定模型和阈值后仍需外部验证。CPU 契约测试、GPU 可运行性、完整 P/R 分开记录。

这条路线是待验证的试验，不能将源码存在或 CPU 测试通过当成精度提升。MinneApple 整图切片融合评估尚未实现，本轮模型选择只使用 PApple VAL。

## 换电脑运行

完整交接包私下提供 `assets/` 和 `weights/`。仅克隆 Git 不包含这些大文件。配置为 `configs/pilot.json`，路径相对包根目录，与盘符无关；支持目标电脑只有 C 盘。

依次运行 `01_setup.cmd`、`02_check.cmd`、`03_gpu_stress.cmd`、`04_train.cmd`，完成或失败后运行 `05_feedback.cmd`。恢复正式中断任务使用 `04_resume.cmd`。具体说明见 `先读我_另一台电脑运行.txt`。

检查脚本会核验 CUDA 和资产；GPU 检查执行真实 768 输入的两个成功更新，不输出精度结论。正式检查点仅在完整更新边界保存，恢复时绑定源码、配方、数据身份、采样顺序和随机状态。

目标电脑需 Python 3.11。环境、缓存和临时目录均在包内；安装脚本从 [PyTorch 官方 cu132 索引](https://download.pytorch.org/whl/cu132/torch/) 安装已列出的 Windows 构建。实际驱动兼容性和 RTX 3080 执行情况由目标电脑预检确认。

## 环境与数据

当前实测环境为 Windows、Python 3.11.15、PyTorch 2.12.1+cu132、Transformers 4.57.1。依赖列在 `requirements.txt`，实测 CUDA 构建列在 `constraints-tested.txt`。PyTorch/torchvision 的 CUDA 构建需要与机器驱动匹配，参考 [PyTorch 官方安装页](https://pytorch.org/get-started/locally/)。新实现沿用现有 F 盘环境进行测试。

数据和大文件不进入 Git。公开来源为 [PApple v4](https://zenodo.org/records/7260694) 和 [MinneApple](https://github.com/nicolaihaeni/MinneApple)。仓库只记录来源、配置和哈希；本机路径写入被忽略的 `configs/local.json`。官方测试数据不参与训练或阈值选择。

`reference/` 是历史快照，`src/` 是新开发入口。权重、数据集、环境、缓存、完整运行输出和私有对话不通过普通 Git 上传。私有交接包另含脱敏可见文本对话；它不是原始内部推理或工具记录归档。

</details>
