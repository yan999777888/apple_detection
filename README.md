# apple_detection

苹果可见区域／完整果实双分支实例分割研究。当前开发以 Mask2Former 为基础，服务于矮化密植果园中的识别与后续定位。

## 验收与已测基线

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
