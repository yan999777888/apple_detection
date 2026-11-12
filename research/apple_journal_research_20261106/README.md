# 苹果视觉期刊研究工作目录

本任务已按用户要求迁入 F 盘。后续请在本目录执行，并先运行 `. ./set_run_env.ps1`，将任务缓存和临时文件放在本目录的 runtime 下。

- `migration_receipt.json`：3329 个原工作区文件，共 694534700 字节，两次 SHA-256 核验后移除了 C 盘三个任务目录。
- `relocation_complete.json`：11 个运行清单或配置的重定位记录，8179 处文件路径引用已检查。
- `migration_original_text/`：重定位前的清单原文；历史实验报告和 checkpoint 原样保留，其中旧路径通过重定位映射解释。
- `apple_research_audit/`：原模型、数据和 Small+Base 组合测速审计。
- `apple_research_v2/`：公开数据准备、标注队列、原三次试运行与继续训练脚本。
- `assets/`：已核验来源的新预训练权重及许可证记录。
- `experiments/`：后续完整训练、消融和评估输出。

当前继续进行全量公开数据基线训练与开发集验证。成熟度开发集半成熟类仅 3 个实例，且两套公开数据缺少可靠逐果身份，所有相应指标必须标明是开发验证，不能称独立果实测试。原始本地数据没有的成熟度和商品等级标签继续保持未标注。

系统目标为移动底盘、单机械臂、采摘时按视觉等级投放多个箱子。当前数据尚不能直接验证整果商品等级、隐藏缺陷和实际投箱成功率。轻量网络基线只是后续机制实验的参照，不自动构成论文创新。


## 完整实验与当前续跑

- 成熟阶段分割及 SSDA 缺陷检测已各完成固定 40 轮，最终权重、开发集结果及独立核验在 `runs/stage_yolo26n_seg_640_e40_s20261106/` 与 `runs/ssda_yolo26n_detect_640_e40_s20261106/`。主结果使用 epoch40，best-dev 另存。
- 全量 ROI 属性分类 12 个 20 轮实验在 `experiments/attribute_full_results_v3_1.json`；ROI 由真值提供，不代表整图系统成绩。类别加权在本轮对照中没有改善。
- 两个模型的原生/ONNX CUDA 一致性与完整图像流水线计时在 `benchmarks/*_native_onnx_parity_cuda_v1/`。阶段72图、缺陷113图全部通过；计时各40图×3轮，不含相机/深度/跟踪/机械臂。
- `experiments/resolution_pairs/` 保存同权重640/960配对和单独官方AP。提高推理分辨率对两组开发数据整体变差。原配对脚本的可选官方记录分支存在接口错误，失败保留；完整官方重评由独立 `validate_resolution_pairs_official.py` 成功执行。
- `experiments/budget_replay/fixed_budget_replay_results_v1.md` 保存固定预算非学习门控及GT上限。低置信度复核不是已证实改进；离线成本账本不换算系统FPS。
- PApple 2304张训练图/814张开发图的40轮训练已经进入实际 batch。`continue_papple_v1.ps1` 成功训练后自动执行共同8图与40图计时及ONNX导出；`experiments/papple_continuation_v1_events.jsonl`、`runs/papple_yolo26n_seg_640_e40_s20261106/status.json` 和 `logs/papple_continuation_v1_train40.log` 是实时状态来源。不会自动重启、续训或覆盖已有run；PApple ONNX推理一致性尚未排入该队列。
- HDF 5000张作者real图已验证保存，合成图未取。去重、跨类别同源家族隔离及代理分组在 `apple_research_v2/sources/additional_attribute_sources/hdf_real_only/`。图像级品质诊断只用于识别偏差，不视为独立果实分级验证。

所有新增输出及任务缓存位于本 F 盘研究目录。后续应继续构建按来源缺失监督的共享主干计算对照与可补救性机制；需要真实逐果/场景隔离标签和单臂投箱试验才能支持完整系统及经济效益主张。

最新完整交接快照：`experiments/parent_handoff_20261106T140852Z.json`。HDF外围诊断及独立核验已完成；共享主干原型v1_1通过CPU真实损失/梯度测试，但没有联合训练或实测共享模型速度。PApple仍按实时状态文件续跑。
