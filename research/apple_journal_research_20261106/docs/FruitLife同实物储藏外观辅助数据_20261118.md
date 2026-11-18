# FruitLife 同实物储藏外观辅助数据

从 Mohamed Sediq Azarkhordad 与 Zobeir Raisi 发布的 FruitLife v1.5 中提取 360 张主采集苹果照片。六颗苹果各有 60 张连续观测。每种储藏条件（日晒、室内、阴凉）有两颗独立实物，均经历作者阶段 0 至 4。图像有色卡和尺，原始阶段是人工判断的外观变化。CC BY 4.0。[作者数据与许可](https://doi.org/10.5281/zenodo.23589223)

采用作者的盲审修订清单 `manifest_v2_b6consensus.csv`。多数原标签由一人评定，一部分转折帧经过双人盲审及裁决。苹果专属标注说明见作者压缩包 `fruitlife_code_v1.5.tar.gz` 内的 `fruitlife_code/docs/b6-transition-relabel-instructions.md`。阶段 0 描述有光泽、表面平滑、保持品种原色；阶段 1 可以只是光泽消退，也可以已有褐变，果实仍保持圆形且表面平滑。阶段 2 出现细皱纹，阶段 3 有明显萎缩或局部塌陷，阶段 4 大部分果皮褐变塌陷并带有深皱。阶段 1 用于训练时仅提供轻度储藏外观的弱监督。阶段 0 也可能带有原本的果脐纹理或小外观痕迹。

打开模型输出前，在每个环境内，使用固定字符串与实物 ID 的 SHA-256 顺序各取一颗训练、另一颗永久留出。同一颗的 60 张照片全部在同侧。[固定实物分配](../datasets/public_defect_intake/fruitlife_v1/physical_id_split.json)

| 用途 | 实物 ID | 阶段 0 | 阶段 1 | 阶段 2 | 阶段 3 | 阶段 4 |
|---|---|---:|---:|---:|---:|---:|
| 可用于训练的实物 | 02、04、06 | 15 | 15 | 15 | 86 | 49 |
| 永久外部复核的实物 | 01、03、05 | 29 | 21 | 15 | 71 | 44 |

仅将训练侧阶段 0 和 1 的 30 张照片加入已有的 Woo 58 张可见表面辅助训练样本。其余 150 张训练侧照片保留本地，不用于本轮训练。外部的 180 帧按五级分别汇报。该外部集只有三个独立实物，每组的图像计数不等同于实物数量。它检验随储藏发生的外观变化，不能替代面向损伤、虫伤和划痕的独立商品等级真值。[训练清单](../datasets/apple_matched_fruitlife_woo_v1/manifest.json) · [外部复核清单](../datasets/apple_matched_fruitlife_woo_v1/external_test.json)

使用作者提供的去阴影果实掩膜，按 JPEG 原始像素（忽略手机 EXIF 的显示旋转）配准，沿果框向外扩展 10%，掩膜以外统一设为灰色，避免引入色卡、刻度尺和实验板。检查每张的图像尺寸、掩膜尺寸、表面非空及文件指纹。按每个实物、每个阶段的时间中位帧查看了 30 个裁片，均可见完整果体。日晒实物的阶段 1 可已有较明显的整果褐色，室内实物的阶段 0 和 1 在图上差别可以很小。[训练阶段照片](../datasets/public_defect_intake/fruitlife_v1/training_stage_contact.jpg) · [外部裁片配准检查](../datasets/public_defect_intake/fruitlife_v1/external_crop_qc.jpg)

原始图像归档为 2,965,994,476 字节，读取其连续压缩块至全部苹果被提取，故未重新校验全物种归档的完整 MD5。实际取用的作者清单与掩膜包核验了官方 MD5，360 份苹果原图和裁片逐个计算 SHA-256。与原 AFruitDB、Woo 裁片和 HDF 代表图的 SHA-256 比较没有字节重复。[可重现下载及裁片代码](../apple_research_v2/fruitlife_apple_source.py) · [构造清单与抽检统计](../apple_research_v2/fruitlife_matched_grade.py) · [冻结清单的校验值](../datasets/apple_matched_fruitlife_woo_v1/source_audit.json)
