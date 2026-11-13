# 下一代 BERT：优化方案与缩减验证

仓库包含实验报告、代码、固定处理数据以及三个随机种子的正式结果。复现实验命令见下文。

目标模型配置为 **290,199,552 参数、最大输入长度 8192**。实际训练的是 **6,921,216 参数**的缩减模型，训练长度为 256；8192 token 仅用于前向执行和人工构造的长程诊断。完整规模模型未进行预训练，实验也未与现有编码器完成同预算比较。

## 文件说明

| 路径 | 内容 |
| --- | --- |
| LLM_HW2.4_Next_BERT_Experiment_Report.md | 缩减验证实验报告 |
| src/scaled_model.py | 原生 PyTorch 缩减／完整结构 |
| src/scaled_data.py | 文档划分、训练词表、哈希与数据审计 |
| scripts/run_scaled.py | 三种子训练、四分支、四类任务与资源评测 |
| scripts/check_target.py | 完整规模随机初始化前向检查 |
| scripts/lexical_baselines.py | 朴素贝叶斯、BM25 和随机参照 |
| scripts/build_figures.py | 从正式 JSON 生成报告中的 SVG 可视化 |
| scripts/audit_results.py | 根据固定测试标签和逐例预测复算正式指标 |
| scripts/fetch_data.py | 按来源和哈希重新获取原始数据 |
| configs/optimized_experiments.json | 实验配置说明 |
| data/scaled | 固定处理数据、8192 词表及 SHA256 清单 |
| results/validation | 正式训练记录、指标、逐例预测及环境 |
| data/raw（仅本地） | 原始下载数据，用于核验来源和重新准备数据 |
| results/validation/seed_*/*.pt（仅本地） | 编码器权重和分类、抽取、重排任务头 |
| results/figures | 报告使用的任务、长程证据和部署成本 SVG 图 |
| tests | 注意力数值／梯度、数据分离、分词和掩码测试 |

## 环境与依赖

已运行环境为 Windows、Python 3.12.4、PyTorch 2.5.1+cu121。最终流程不需要下载预训练编码器，也不依赖 transformers、pandas 或 TensorFlow。

可在独立环境中安装 CUDA 版 PyTorch 及项目依赖：

```powershell
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt
$env:PYTHONIOENCODING='utf-8'
```

## 用仓库内的固定数据复现

仓库已包含固定处理数据和 tokenizer，可离线训练。从项目根目录执行：

```powershell
python -m pytest tests -q
python scripts/run_scaled.py --seeds 17 42 71 --base-steps 3000 --branch-steps 500 --output results/reproduction
python scripts/lexical_baselines.py
python scripts/check_target.py
```

训练输出目录为 `results/reproduction/`，与正式结果分开保存。中断后可使用同一目录继续运行；脚本会跳过已有检查点和评测结果，并校验检查点对应的阶段与训练步数。更改配置时使用新的输出目录。

固定种子不保证不同显卡、驱动和 PyTorch 版本逐位一致。完整规模检查仅做无梯度编码器前向，不建立优化器，不计算全词表 logits。

正式指标保存在 `results/validation/`。以下命令可重新生成图表并复核指标：

```powershell
python scripts/build_figures.py
python scripts/audit_results.py
```

## 从原始数据重建

```powershell
python scripts/fetch_data.py
python -m src.scaled_data
```

下载脚本按已保存清单校验哈希；上游内容改变时会报错。重新准备数据会重新训练 BPE，跨版本可能改变 token ID，故优先使用仓库内冻结数据和词表。重建后必须重新训练模型，不能继续使用旧词表权重。

## 实验范围

每个种子先训练共同起点 3000 步，再训练 E0～E3 各 500 步。E0 为 token MLM；E1 在一半步骤使用片段掩码；E2 将窗口从 64 扩至 128；E3 在相同双视图 MLM 计算基础上增加对比项。每组均评估中英文分类、抽取式问答、检索和条件重排。

本轮实验比较了片段掩码、局部窗口扩大和双视图对比适配，并评估中英文分类、抽取式问答、检索、重排、长程证据利用及部署延迟。词表规模、归一化、FFN 类型、RTD、文本扩散和预训练检查点适配未纳入实验。

训练和指标复核使用已冻结的 `data/scaled/`，无需重新下载原始数据。独立复现实验可保存到 `results/reproduction/`。
