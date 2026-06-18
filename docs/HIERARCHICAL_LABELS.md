# Hierarchical Signal Labels

本文档说明本数据集中的“大类”和“小类”标签关系。数据集默认的主要任务是 17 类 HF digital-mode 细粒度分类；同时，这 17 个小类也可以整理成若干个粗粒度信号家族，用于大类分类、层级分类和错误分析。

## Label Types

当前生成的数据集中包含三类主要标签：

| File | Meaning | Is it a signal class? |
|---|---|---|
| `y.npy` | 细粒度信号模式标签，例如 `BPSK31`、`Olivia_4_250`、`RTTY` | Yes |
| `snr_db.npy` | 每个样本的信噪比标签，例如 `25`、`10`、`0`、`-10` dB | No |
| `channel_profile.npy` | 每个样本使用的信道模型/衰落剖面标签 | No |

因此，`SNR` 和 `channel_profile` 不是信号类别，而是样本对应的接收条件。真正的信号类别是 `y.npy`。如果需要层级分类，可以根据 `mode_names.json` 和 `y.npy` 额外派生 `family_label.npy`。

## Fine-Grained Classes

默认数据集包含 17 个细粒度信号小类：

| Fine ID | Fine class |
|---:|---|
| 0 | `BPSK125` |
| 1 | `BPSK31` |
| 2 | `BPSK63` |
| 3 | `Contestia_4_250` |
| 4 | `Contestia_8_500` |
| 5 | `CW` |
| 6 | `DominoEX_8` |
| 7 | `MFSK16` |
| 8 | `MFSK32` |
| 9 | `Olivia_4_250` |
| 10 | `Olivia_8_500` |
| 11 | `QPSK125` |
| 12 | `QPSK31` |
| 13 | `QPSK63` |
| 14 | `RTTY` |
| 15 | `Thor_16` |
| 16 | `Throb_2` |

这些标签对应机器学习中的主分类目标，也就是通常所说的 17 类信号识别任务。

## Coarse Families

为了更符合实际短波信号识别流程，可以把 17 个小类进一步归纳为 5 个粗粒度大类：

| Coarse ID | Coarse family | Fine classes |
|---:|---|---|
| 0 | `PSK` | `BPSK31`, `BPSK63`, `BPSK125`, `QPSK31`, `QPSK63`, `QPSK125` |
| 1 | `Olivia_Contestia` | `Olivia_4_250`, `Olivia_8_500`, `Contestia_4_250`, `Contestia_8_500` |
| 2 | `MFSK_like` | `MFSK16`, `MFSK32`, `DominoEX_8`, `Thor_16`, `Throb_2` |
| 3 | `RTTY_FSK` | `RTTY` |
| 4 | `CW` | `CW` |

这里的 `Olivia_Contestia` 和 `MFSK_like` 是面向实验的分组方式。Olivia 和 Contestia 在调制结构上与多音/MFSK 类信号有关系，但由于它们具有明显的 tones/bandwidth 参数组合，并且在模型结果中容易形成一个相互混淆的子群，因此单独作为一个粗粒度家族更便于分析。

## Recommended Classification Tasks

本数据集可以支持三种常见任务：

1. 细粒度分类：直接预测 17 个具体信号模式，即使用 `y.npy`。
2. 粗粒度分类：预测 5 个信号家族，即使用派生的 `family_label.npy`。
3. 层级分类：先预测粗粒度家族，再在该家族内部预测具体小类。

在论文中建议这样描述：

> The dataset contains 17 fine-grained HF digital-mode classes organized into five coarse signal families, including PSK, Olivia/Contestia, MFSK-like modes, RTTY/FSK, and CW. The primary label is the fine-grained mode label, while optional coarse family labels can be derived from the mode metadata for hierarchical classification and error analysis.

中文表述可以写为：

> 本数据集包含 17 个细粒度 HF 数字模式类别，并可进一步归纳为 PSK、Olivia/Contestia、MFSK-like、RTTY/FSK 和 CW 五个粗粒度信号家族。默认分类标签为具体模式标签，研究者也可以根据模式元数据派生大类标签，用于层级分类和混淆分析。

## Generate Optional Family Labels

如果需要在数据集中显式保存大类标签，可以在生成好的数据集目录中运行下面的 Python 片段：

```python
from pathlib import Path
import json
import numpy as np

dataset_root = Path(r"E:\仿真\panoradio_like_ccir520_npy_fixed")

mode_names = json.loads((dataset_root / "mode_names.json").read_text(encoding="utf-8"))
y = np.load(dataset_root / "y.npy", mmap_mode="r")

family_names = [
    "PSK",
    "Olivia_Contestia",
    "MFSK_like",
    "RTTY_FSK",
    "CW",
]

family_by_mode = {
    "BPSK31": "PSK",
    "BPSK63": "PSK",
    "BPSK125": "PSK",
    "QPSK31": "PSK",
    "QPSK63": "PSK",
    "QPSK125": "PSK",
    "Olivia_4_250": "Olivia_Contestia",
    "Olivia_8_500": "Olivia_Contestia",
    "Contestia_4_250": "Olivia_Contestia",
    "Contestia_8_500": "Olivia_Contestia",
    "MFSK16": "MFSK_like",
    "MFSK32": "MFSK_like",
    "DominoEX_8": "MFSK_like",
    "Thor_16": "MFSK_like",
    "Throb_2": "MFSK_like",
    "RTTY": "RTTY_FSK",
    "CW": "CW",
}

fine_to_family = np.array(
    [family_names.index(family_by_mode[name]) for name in mode_names],
    dtype=np.int16,
)

family_label = fine_to_family[y]
np.save(dataset_root / "family_label.npy", family_label)
(dataset_root / "family_names.json").write_text(
    json.dumps(family_names, indent=2),
    encoding="utf-8",
)
```

生成后会多出两个文件：

| File | Type | Description |
|---|---|---|
| `family_label.npy` | `int16`, shape `[N]` | 每个样本对应的大类标签 |
| `family_names.json` | JSON list | 大类名称列表 |

## How to Use in Experiments

细粒度分类时，模型输出类别数应设为 17：

```text
num_classes = len(mode_names)
target = y
```

粗粒度分类时，模型输出类别数应设为 5：

```text
num_classes = len(family_names)
target = family_label
```

层级分类时，可以训练两个模型，也可以训练一个带有两个输出头的模型：

| Output head | Target | Purpose |
|---|---|---|
| Coarse head | `family_label` | 判断信号大类 |
| Fine head | `y` | 判断具体模式 |

这种设置有助于分析模型到底是“大类判断错了”，还是“大类正确但小类混淆”。例如，`Olivia_8_500` 和 `Contestia_8_500` 在短向量、低信噪比和衰落条件下容易互相混淆；层级标签可以更清楚地显示这种错误属于同一粗粒度家族内部的细分类错误。

## Suggested Paper Wording

可以在论文摘要或数据集说明中加入：

```text
In addition to the 17 fine-grained mode labels, the dataset can be organized into five coarse signal families: PSK, Olivia/Contestia, MFSK-like modes, RTTY/FSK, and CW. This hierarchical label structure supports both conventional fine-grained signal classification and coarse-to-fine recognition experiments.
```

如果正文中需要更正式一点，可以写：

```text
The label space is hierarchical. The primary label corresponds to the fine-grained digital mode, while a deterministic mapping from mode names to coarse signal families is provided for hierarchical evaluation. This enables three benchmark settings: coarse-family classification, fine-grained mode classification, and coarse-to-fine hierarchical classification.
```
