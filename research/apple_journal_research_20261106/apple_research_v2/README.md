# 苹果视觉研究：公开数据准备与属性训练工程基线

日期：2026-11-06。接续原任务与 `apple_research_audit/README.md` 的本机审计。

当前目标遵循用户最新要求：移动底盘＋单机械臂，采摘时判断哪些果实值得摘、并直接按视觉等级投放底盘多个箱子；不要求双臂或额外采后分拣线。本目录完成数据和训练通路建设，不直接控制机械臂，不替换原Small＋Base权重。

## 已完成的工作

- 复核用户F盘原数据：1918张训练原图、306张验证原图；实际标签为单类苹果框，无成熟度或商品等级。原数据与权重只读保留。
- 实际下载两份公开CC BY 4.0归档，合计450,728,360字节，校验发布方的大小和摘要；原ZIP及来源metadata保留。
- 完成Mendeley原图家族/增强/重复图隔离，以及SSDA采集日期分组与异常框隔离。
- 实际执行三次32步GPU训练试运行（两项基线与一次加权对照），保存真实线性头、逐步loss、优化器计数、输入/模型哈希与ROI测速。
- 从用户原训练图生成60张待人工标注裁片；属性字段全为null，没有自动编造标签。
- 34项单元测试通过，覆盖标签缺失拒绝、分组泄漏、test访问拒绝、数值标签原义、计时和类别加权；数据准备另有实际文件/标签/哈希核验。

所有新增材料位于当前Codex工作区；未修改F盘原数据/模型，未安装或升级共享环境，未提交Git远端。

## 数据来源、许可和真实含义

| 来源 | 已核验的用途 | 当前使用范围 |
|---|---|---|
| [Wang & Wang，Mendeley v4](https://data.mendeley.com/datasets/gfcmdbvw65/4)，DOI 10.17632/gfcmdbvw65.4，CC BY 4.0 | 作者依据可见颜色标注未成熟/半成熟/成熟阶段；不等价于糖度、硬度或生理成熟测量 | 原分辨率train/val及作者标注；不将两种分辨率算作不同图 |
| [Nguyen，SSDA v1.0.0](https://zenodo.org/records/10957715)，DOI 10.5281/zenodo.10957715，CC BY 4.0 | 苹果表面缺陷区域的两个数值类别；没有正常果或商品等级标签 | 室内黑背景、1024×1024图像；用于缺陷区域工程基线 |
| [Fuji-Ripeness-Size](https://www.kaggle.com/datasets/zhukeyi1/fuji-ripeness-and-size-dataset) | 包含PApple/AmodalAppleSize来源及另一来源；平台标示GPL3 | 只保存公开metadata/文件索引，未下载大数据、未训练；需保持底层来源许可与身份关系 |

Fuji索引含8059文件，展开文件列表总约62.36GB；其中2304张train及814张val文件名对应本机PApple的既有划分。这里只核对文件名，不等于逐字节同源核验，但足以阻止将其当作未见过的独立外部测试。另一个API的15.13GB数值与文件索引统计口径不同，均保留原始metadata，不混称总展开体积。

Mendeley下载文件为290,131,787字节，SHA256 `049591afd4fc3529aedbd31ef9119f5ec0601ebcb8cbbd4a25b7e883fde147ce`。SSDA为160,596,573字节，发布方MD5 `86ad2aad9cbe1bdba8b85814a2cfe8c8`，本地SHA256 `9d98ee39e3c7dffccdc42ab11606d88002b35be316c44da1ed08d99830f9f0fc`。对应receipt位于 `downloads/`。每份准备目录保留出处和修改说明，不去除署名。

## 实际划分与清理

### Mendeley

采用 `prepared/mendeley_v4_original_documented_v2/`，训练输入是 `adapter_manifest.json`。原训练标注1041图/1981实例，验证126图/185实例。最终保留：

| 分区 | 图像 | immature | semi-mature | mature | 总实例 |
|---|---:|---:|---:|---:|---:|
| 工程训练 | 772 | 711 | 276 | 545 | 1532 |
| 仅保留的开发数据，未参与本轮训练/评价 | 72 | 65 | 3 | 46 | 114 |

排除train244/val43个预生成brightness/noise增强图；按去增强后原图家族检查交叉，排除train24/val10张交叉家族原图；另排除跨集合完全相同字节的train/398.jpg及val/166.jpg。任何在test文件名列表出现的家族也不得进入训练。只用原分辨率，不重复加入`*_resize`。

作者README明确polyline实际代表闭合多边形，因此允许隐式闭合；没有改点、改类别或填补轮廓。早期`original_strict_v1`按首末点相等过滤过严，已弃用，不能用于训练。本轮bbox取原坐标范围，不能把作者轮廓自动视为严格可见区域V真值。

仍没有可靠fruit/tree/scene-ID或EXIF。家族及重复图隔离不证明果实独立，半成熟开发实例仅3个；因此本轮trainer明确跳过验证，`validation=null`。数据准备脚本不读取test内容；初查ZIP时曾读过一次公开test标注模式示例，如实保留此记录，test从未用于训练、评估或阈值选择。原PApple封存test没有用于本轮训练或评价。

### SSDA

采用 `datasets/ssda_date_v2/`。归档实际含710张1024×1024 RGB图片，不是网页描述的原始传感器分辨率。全部图片和标签共1420文件验证ZIP CRC、Git blob及落盘SHA，未执行下载仓库代码或加载其缓存。

原568/142划分有45个同秒不同相机采集组跨集合，不能直接使用其随机划分证明同果独立。本轮改为按日分组：

| 分区 | 日期 | 图像 | class_0 | class_1 |
|---|---|---:|---:|---:|
| 工程训练 | 20211104 | 596 | 537 | 739 |
| 仅保留的开发数据 | 20211025、20211027 | 113 | 87 | 143 |
| 隔离 | 20211104_133557采集组 | 1 | 0 | 6 |

源标签一框下边界越出2.1363像素，超出四舍五入误差。该图及同秒组从实验清单整体隔离，6个框均不用于训练，原始字节保留，没有裁剪框或把遗漏框当背景负例。失败的v1及对应脚本作为审计证据保留。

两类别暂无可靠数字→中文类型映射文件，使用`ssda_class_0/ssda_class_1`，不凭外观猜语义。全部源图都有缺陷标注，不能由此生成“正常/优级”类。日期隔离不证明物理果实独立，也不证明室内图像能泛化到树上场景；本轮同样跳过验证。

## 已执行训练

使用本机已核验的官方COCO Mask R-CNN ResNet50初始化，SHA256 `bf2d0c1efbc936eeee2bc95a48e80ebc86b891f61b0106485937fc29f9315fc0`；不是已经接触苹果数据的专项权重。严格恢复265个`backbone.body`状态，FrozenBatchNorm，骨干eval且无梯度，全球池化得到2048维特征，仅训练新Linear头。

两项基础工程运行均固定seed20261106、FP32、batch4、AdamW、lr0.001、weight_decay0.0001、32次实际更新。每项128次ROI训练曝光，没有声称已遍历全部数据或收敛。缺失标签、test记录、越界ROI、源图/组跨集合、输入哈希改变都会拒绝；没有静默跳batch。

| 工程基线 | 固定训练probe平均CE：前→后 | ROI平均/P95耗时 | 权重 |
|---|---:|---:|---|
| 外观阶段3类 | 1.090445→1.003825（24个train ROI） | 11.115 / 15.233 ms | `runs/mendeley_stage_linear32_seed20261106_v1/linear_head_pilot.pt` |
| 缺陷区域2类 | 0.699097→0.502843（16个train ROI） | 9.367 / 12.095 ms | `runs/ssda_defect_region_linear32_seed20261106_v1/linear_head_pilot.pt` |

负结果完整保留：外观阶段中的semi-mature训练probe CE从0.865997升至1.849400，另两类下降。不能只报告总体loss下降并宣称三类识别改善。固定train probe只检查同样本拟合，没有报告准确率，不是独立评价，也没有用于延长原run。

### 成熟度类别加权的单次探索性对照

基线暴露半成熟类别问题后，另存 `train_attribute_pilot_v2.py`，只改变训练CE的类别权重，按训练标签数计算 N/(C×n)：未成熟0.718237、半成熟1.850242、成熟0.937003。新运行固定同一seed、初始化、32步样本顺序、批次、优化器与训练probe。它是看到基线结果之后提出的探索性对照，不是预注册结果，也没有自动替换原模型。

| 同一24个训练ROI的未加权CE | 初始化 | 原始基线训练后 | 加权训练后 |
|---|---:|---:|---:|
| 平均 | 1.090445 | 1.003825 | 0.917308 |
| 未成熟 | 1.122774 | 0.335048 | 0.574252 |
| 半成熟 | 0.865997 | 1.849400 | 1.458603 |
| 成熟 | 1.282564 | 0.827028 | 0.719069 |

半成熟相对原基线有所改善，但仍比初始化差；未成熟相对原基线退步。不能由这一小批训练probe声称成熟度识别问题已经解决。加权与未加权的训练批次loss不可直接横比，表格统一使用未加权probe CE。没有开展后续搜索或扩展训练。

对照报告为 `runs/mendeley_weighting_comparison_v1.json`，新权重为 `runs/mendeley_stage_weighted32_seed20261106_v2/linear_head_pilot.pt`。原基线报告按生成时状态原样保留；其中“未用于调参”描述当次运行，后来使用其负面结果提出本次对照的事实在比较报告单独记录。

GPU运行后CPU独立复核实际checkpoint：原两项基线的weight/bias优化器step均32，头参数每步真实变化，骨干及全部输入哈希前后一致。第三项加权运行同样记录32次真实更新，并在比较报告核对相同初始化、样本顺序和输入哈希。报告存每步真实loss与记录索引。头checkpoint需配合上述固定COCO骨干加载；并非完整的机器人部署模型。

本次分类器使用标注提供的真实ROI，不包含检测器预测框误差，不能视作端到端检测或分级性能。ROI测速从预解码RGB开始，包含ROI裁剪、224缩放、归一化、H2D、ResNet＋头和CPU概率，8个ROI×3轮，预热3次。单位是ROI/s，不是全局检测FPS；排除检测器、深度、跟踪、相机与投箱控制。每果实多个属性/多目标的总体成本尚未评测，不能将约90/107 ROI/s写成整机速度。

## 用户原数据的待标注队列

`data/annotation_queue_v1/annotation_queue_v1.jsonl`及`crops/`保存60个真实果实预览：LOCAL_apple18、LOCAL_shuju42，60张不同原图，覆盖42个已有训练采集组、每组最多2框；只使用原始train图，不用tile、增强或val/test。保存原图与标签路径、SHA、原框、裁片坐标和manifest索引。

`maturity`、`visible_defects`、`grade`、`fruitID`全部为null。现有时间块/编号块不是经核实果实身份，60张图不能称60个独立果实。部分小目标分辨率有限，应允许人工填“不可判定”，而不是强行定级。

## 下一阶段的可检验方案

1. 首先确定采摘品种与采摘时可观测的外观等级、对应箱号、误分级代价、不可判定处理。当前没有这套真实等级标签，不能由公开三阶段标签或缺陷类别直接推导商品等级。
2. 以轻量全图检测/可见分割和共享特征多任务头作为基线，旧Small＋Base保留对照/离线教师。各来源只监督其真实存在的任务；其他任务loss屏蔽，不把缺失缺陷标签当健康或缺失成熟度当成熟。
3. 创新假设仍是任务专属证据充分度与受时限ROI复查：在相同覆盖率、同硬件/总时延预算下，是否降低重遮挡的属性错误接受率。普通多任务、单检测分数门控、可见比例门控、固定top-K复查与收益/成本复查必须对照。当前工程线性头不是该创新已经成立的证据。
4. 数据采集按真实果实/完整作业序列/树/日期分组，同果不同视角不跨集合。树上选果与接近/摘取/转运中补视角共同记录fruit-ID，投箱前确定等级；末端相机或换视角是可选辅助，不依赖额外分拣线。遮挡未见表面不能由单帧RGB判定无缺陷，待定箱也只是可选规则。
5. 正式训练须在有效标签、划分和固定预算确定后执行，至少比较多随机种子及按果实/序列聚类的区间。评价需要检测/分割、属性macro-F1/有序误差、拒判覆盖率、错投箱率、整帧P95/帧龄、单果周期，以及按实际等级价格和损耗计算的经济结果。

源代码、数据清单和两项工程基线已可重复运行；缺少可靠独立身份与真实等级标签仍限制正式性能结论。本轮没有宣布可投稿达标、独立准确率提升或自动分类投箱已就绪。

## 复现入口

在任务根目录PowerShell执行，所有输出使用新的目录名，避免覆盖证据：

```powershell
$python = 'F:\研究生毕业设计\模块一定位补偿以及预测\apple_detection\envs\local_rgbd_v1\Scripts\python.exe'
$env:PYTHONDONTWRITEBYTECODE = '1'
& $python -m unittest discover -s tests -v
& $python apple_research_v2/train_attribute_pilot.py --manifest apple_research_v2/prepared/mendeley_v4_original_documented_v2/adapter_manifest.json --output apple_research_v2/runs/new_stage_pilot --check-only
& $python apple_research_v2/train_attribute_pilot.py --manifest apple_research_v2/prepared/mendeley_v4_original_documented_v2/adapter_manifest.json --output apple_research_v2/runs/new_stage_pilot --max-steps 32
& $python apple_research_v2/train_attribute_pilot.py --manifest apple_research_v2/datasets/ssda_date_v2/attribute_train_manifest.json --output apple_research_v2/runs/new_defect_pilot --max-steps 32
& $python apple_research_v2/train_attribute_pilot_v2.py --manifest apple_research_v2/prepared/mendeley_v4_original_documented_v2/adapter_manifest.json --output apple_research_v2/runs/new_weighted_stage_pilot --class-weighting inverse-frequency --max-steps 32
```

数据准备脚本为`download_verified.py`、`prepare_mendeley.py`、`adapt_mendeley_manifest.py`、`prepare_ssda.py`和`make_annotation_queue.py`，各自`--help`或源码说明参数。SSDA检测训练YAML位于`datasets/ssda_date_v2/data.yaml`，这里只准备好检测入口，本轮未运行缺陷检测器训练，不能把region分类试跑称为检测mAP实验。
