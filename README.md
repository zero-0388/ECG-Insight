# ECG-Insight：心电图多标签识别与案例分析

智能医学工程个人学习项目。基于 PTB-XL 数据，比较小型 1D-CNN 与 1D-ResNet，分析五个诊断大类，并通过 Streamlit 展示公开样本、CPU 模型推理和已保存的积分梯度分析。

## 当前功能

- 7 条精选公开案例，12 导联波形及真实标签展示。
- CPU 实时推理与 Kaggle 保存预测切换，显示逐类数值差异。
- 固定验证集阈值、多标签预测与结果 CSV 下载。
- 两个 HYP 案例的积分梯度热图、统一色标对照和 V6 扰动分析。
- 分析报告展示与下载。

目前不支持上传任意心电图、照片识别或在线重新计算归因。精选案例不能用于估计模型总体性能。

## 本地启动

建议 Python 3.12。Windows PowerShell：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

打开 http://localhost:8501 。首次启动的邮箱提示可以直接回车跳过。该地址只能在本机访问，当前项目尚未部署到公网。

模型文件需位于 `models/best_model.pt`，样本包需位于 `data/demo_samples.npz`。样本波形已标准化，推理时不要再次标准化。`inference.py` 从受信任的项目权重加载模型，并校验类别顺序与输入形状。

## 实验配置与结果

筛选 21,388 条有诊断大类的记录。按官方 `strat_fold` 使用 1–8 折训练、9 折验证、10 折测试，并检查患者无重叠。

| 集合 | 记录数 | 患者数 |
|---|---:|---:|
| 训练 | 17,084 | 14,823 |
| 验证 | 2,146 | 1,917 |
| 测试 | 2,158 | 1,877 |

100 Hz、10 秒、12 导联；使用训练集每导联均值和标准差处理所有集合。类别顺序固定为 NORM、MI、STTC、CD、HYP。HYP 包含心房负荷增大／扩大等子类，不等同于仅预测心室肥厚。

两种模型使用种子 42、批大小 128、Adam 学习率 0.001、无加权 BCEWithLogitsLoss，训练 10 轮，以验证 Macro AUROC 选择最佳权重。每类阈值仅在验证集按 F1 选择。

| 模型 | 参数量 | 验证 Macro AUROC | 测试 Macro AUROC | 测试 Macro F1（阈值 0.5） | 测试 Macro F1（验证集阈值） |
|---|---:|---:|---:|---:|---:|
| 1D-CNN | 55,205 | 0.9174 | 0.9106 | 0.6978 | 0.7277 |
| 1D-ResNet | 178,757 | 0.9294 | 0.9264 | 0.7334 | 0.7473 |

这是单次训练结果，未报告多随机种子重复或置信区间。模型容量不同，不能把提升全部归因于残差连接。测试结果已被查看，后续开发应避免据此调参；新增方案用验证集选择，并记录测试使用情况。AUROC 不等于准确率。

## 数据来源与一致性边界

- [PTB-XL 官方 1.0.3](https://physionet.org/content/ptb-xl/1.0.3/)：标签、元数据、患者划分。
- [Kaggle PTB-XL 镜像](https://www.kaggle.com/datasets/khyeh0719/ptb-xl-dataset)：1.0.1 波形文件。

波形仅按筛选出的官方 1.0.3 记录路径读取；已检查文件覆盖、全部输入的形状／采样率／导联顺序／有限值，以及一条记录与官方波形一致。尚未完成全量文件哈希或版本差异核验，因此结果应标明这套混合来源配置，不能宣称严格复现官方 1.0.3 基准。

PTB-XL 使用 CC BY 4.0；分享导出的案例时应保留数据作者、版本与来源，并说明标准化、筛选及图像生成等修改。数据引用：Wagner 等，PTB-XL，PhysioNet，2022，版本 1.0.3，DOI: 10.13026/kfzx-aw45；原始论文 DOI: 10.1038/s41597-020-0495-6。

参考学习资源：[PTB-XL 论文基准代码](https://github.com/helme/ecg_ptbxl_benchmarking)、[WFDB](https://github.com/MIT-LCP/wfdb-python)、[Captum](https://github.com/meta-pytorch/captum)。当前小型网络并非论文模型完整复现；若进一步复制第三方代码，需要保留其对应许可证。

## 归因与失败分析

积分梯度解释 HYP logit，使用标准化空间全零参考、128 积分步。案例 1745 为漏报，1320 为多标签真阳性，完整性相对残差分别约 0.14% 与 0.32%。小残差只反映积分近似，不验证医学解释。

记录 1320 仅 V6 经二阶 0.5 Hz 高通后，HYP 分数从 0.6579 变为 0.6754，阳性判断不变。此扰动也改变低频波形，不能证明普遍抗噪稳健性。热图不代表病灶位置，预测分数未经概率校准。

## 文件与训练过程

- `app.py`：演示界面。
- `inference.py`：模型结构、权重加载、CPU 推理。
- `models/best_model.pt`：Kaggle 保存的最佳 ResNet 权重。
- `data/demo_samples.npz`：7 条精选案例、标签、预测、阈值与标准化参数。
- `interpretability/`：归因数组、图片、表格及报告。

训练在 Kaggle 完成，源码已导出到 `notebooks/`，按编号顺序运行。下面的原始链接可能仍为私有，复现时可把本地 Notebook 导入自己的 Kaggle 账号：

1. [数据准备](https://www.kaggle.com/code/zero0388/ecg-insight-01)
2. [CNN](https://www.kaggle.com/code/zero0388/ecg-insight-02-training)
3. [ResNet](https://www.kaggle.com/code/zero0388/ecg-insight-03-resnet)
4. [错误分析](https://www.kaggle.com/code/zero0388/ecg-insight-04-error-analysis)
5. [积分梯度](https://www.kaggle.com/code/zero0388/ecg-insight-05-interpretability)

| 本地 Notebook | Kaggle 输入 | 环境与输出 |
|---|---|---|
| `ecg-insight-01.ipynb` | PTB-XL Kaggle 镜像 | CPU、联网；安装 WFDB；输出 `ecg_prepared/` |
| `ecg-insight-02-training.ipynb` | 01 的已保存输出 | GPU；输出 `cnn_baseline/` |
| `ecg-insight-03-resnet.ipynb` | 01 的已保存输出 | GPU；输出 `resnet_baseline/` |
| `ecg-insight-04-error-analysis.ipynb` | 01 与 03 的已保存输出 | CPU、联网获取官方元数据；输出 `error_analysis/` |
| `ecg-insight-05-interpretability.ipynb` | 01 与 03 的已保存输出 | GPU、联网安装 Captum；输出归因文件及演示 ZIP |

导入 Notebook 后，需要手动确认输入来源、Internet 和 Accelerator 设置，不能仅依赖导出文件中的环境元数据。每阶段保存版本时启用保存输出，再把它挂载到下一阶段。新会话必须按顺序从头执行；标准化单元修改内存中的波形，训练单元会更新权重，因此不要在同一会话反复单独运行它们。

已静态检查 5 份 Notebook 的代码语法（排除 Notebook 安装命令）与输出，没有发现语法错误或保存的报错输出。未在干净的 Kaggle 环境重新执行全流程；训练环境的完整依赖版本仍需补录。本地 `requirements.txt` 对应演示环境，不是完整训练环境锁文件。

本地预测核验：

```powershell
.\.venv\Scripts\python.exe verify_predictions.py
```

7 条演示样本已通过 CPU 与 Kaggle 保存预测的数值一致性检查（`rtol=1e-4`、`atol=1e-5`），固定阈值分类一致；这不替代完整实验复现。

## 待完成

- 干净环境下端到端复现，并补录训练依赖版本。
- 全量波形版本一致性核验。
- 多随机种子重复实验。
- GitHub 发布、公网部署与手机访问检查。

本项目是公开数据学习与研究演示，未完成临床验证。
