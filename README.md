<a id="top"></a>

<div align="center">

<h1>ETTm1 Forecasting</h1>

<p><strong>From transformer load data to an informed forecasting model choice.</strong></p>
<p>Ten models · One shared evaluation protocol · Accuracy, efficiency and robustness</p>

<p>
  <img src="https://img.shields.io/badge/Dataset-ETTm1-0F766E?style=flat-square" alt="Dataset: ETTm1">
  <img src="https://img.shields.io/badge/Forecast-96%20steps-2563EB?style=flat-square" alt="Forecast: 96 steps">
  <img src="https://img.shields.io/badge/Models-10-475569?style=flat-square" alt="Models: 10">
  <img src="https://img.shields.io/badge/Framework-PyTorch-EE4C2C?style=flat-square" alt="Framework: PyTorch">
</p>

<p>
  <a href="README.md"><img src="https://img.shields.io/badge/English-Selected-0F766E?style=for-the-badge" alt="English version"></a>
  <a href="README.zh-CN.md"><img src="https://img.shields.io/badge/简体中文-切换-475569?style=for-the-badge" alt="Switch to Simplified Chinese"></a>
</p>

<p>
  <a href="ettm1_end_to_end.ipynb"><strong>Open the notebook</strong></a> ·
  <a href="#quick-start"><strong>Quick start</strong></a> ·
  <a href="#results"><strong>Benchmark results</strong></a> ·
  <a href="#artifacts"><strong>Explore outputs</strong></a>
</p>

</div>

---

<a id="overview"></a>

## Overview

This project brings the ETTm1 forecasting workflow into one executable [Jupyter notebook](ettm1_end_to_end.ipynb): data validation, temporal feature engineering, model tuning, benchmarking, interpretability, ablation studies and stress testing. It connects model accuracy with training cost, inference latency and input robustness to guide forecasting model selection.

| Best benchmark accuracy | Recommended primary model | Fast resource fallback |
| :--- | :--- | :--- |
| **iTransformer** · MAE **0.353577** | **DLinear** · P95 **0.833 ms** | **Ridge** · P95 **0.309 ms** |
| **46.81%** MAE improvement over Naive | **11.489 s** benchmark training time | **0.793 s** benchmark training time |

![Forecasting accuracy versus mean inference latency, with training time encoded by bubble size](outputs/figures/mae_vs_latencymean_tradeoff_m96.png)

**On this page:** [Quick start](#quick-start) · [Experimental protocol](#protocol) · [Results](#results) · [Experiments](#experiments) · [Artifacts](#artifacts) · [Model selection](#deployment) · [Data and references](#references)

<a id="quick-start"></a>

## Quick start

### Local environment

To match the recorded local environment, use **Python 3.13**, **Git** and a Jupyter kernel in the project environment. The notebook retrieves ETTm1 from `raw.githubusercontent.com` and clones THUML Time-Series-Library from GitHub during execution.

```bash
git clone https://github.com/Zayne-Feng/ETTm-Forecasting.git
cd ETTm-Forecasting
```

Create and activate an environment on Windows PowerShell:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

<details>
<summary><strong>Linux / macOS environment commands</strong></summary>

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

</details>

For an **NVIDIA GPU**, install the CUDA 12.8 PyTorch packages first, followed by the project dependencies:

```bash
python -m pip install -r requirements-gpu-cu128.txt
python -m pip install -r requirements.txt
```

For **CPU execution**, install the project dependencies:

```bash
python -m pip install -r requirements.txt
```

Register the kernel and open Jupyter:

```bash
python -m ipykernel install --user --name ettm --display-name "Python (ETTm)"
python -m jupyter notebook ettm1_end_to_end.ipynb
```

Select **Python (ETTm)**. Run the first code cell to install and verify dependencies, restart the kernel, then choose **Run All**. The notebook selects CUDA when available and otherwise uses CPU.

```bash
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

### Azure ML

1. Upload `ettm1_end_to_end.ipynb` and `requirements.txt` to the same working directory.
2. Select the **Python 3.10 - AzureML** kernel described in the notebook's Azure execution guide.
3. Run the dependency installation cell, restart the kernel, then choose **Run All**.

<details>
<summary><strong>Environment recorded in the saved notebook</strong></summary>

| Component | Recorded value |
| :--- | :--- |
| Python | 3.13.12 |
| PyTorch | 2.11.0+cu128 |
| CUDA | 12.8 |
| GPU | NVIDIA GeForce GTX 1650 Ti |
| NumPy / pandas | 2.4.4 / 3.0.2 |
| scikit-learn / Optuna | 1.8.0 / 4.8.0 |
| SHAP / LIME | 0.51.0 / 0.2.0.1 |

The benchmark table records **468.07 s** of combined model training time. This total sums the `TrainTimeSec` column; tuning, explainability, ablations and stress tests are separate stages.

</details>

<a id="protocol"></a>

## Experimental protocol

| Item | Configuration |
| :--- | :--- |
| Data | **69,680** observations at **15-minute** intervals; 2016-07-01 to 2018-06-26 |
| Physical variables | `HUFL`, `HULL`, `MUFL`, `MULL`, `LUFL`, `LULL`, `OT` |
| Main task | **M-96**: predict all seven variables for the next 96 steps from the previous 96 steps; **24 h → 24 h** |
| Input features | Seven physical variables + ten sine/cosine calendar features = **17 channels** |
| Chronological split | **12 / 4 / 4** blocks of 30-day months; train `[0, 34560)`, validation `[34560, 46080)`, test `[46080, 57600)` |
| Window counts | **34,369** training, **11,425** validation and **11,425** test windows |
| Scaling | `StandardScaler` fitted on training data; validation and test inputs include preceding historical context |
| Accuracy metrics | MAE, MSE and RMSE in the **standardized target space**; inverse-transformed per-variable metrics are also displayed in the notebook |
| Skill score | `100 × (1 − model MAE / Naive MAE)` |
| Latency | Single-window inference; mean, median and P95; comparison threshold **P95 < 50 ms** |
| Seed | **42** |

The notebook checks timestamp continuity, duplicate timestamps, missing values, finite tensors and label boundaries. The saved data check reports zero missing values and complete 15-minute timestamp continuity.

### Models and implementations

| Family | Models | Implementation |
| :--- | :--- | :--- |
| Baselines | Naive, Ridge, KNN | Last-observation repeat; scikit-learn regressors |
| Recurrent networks | LSTM, GRU | Local PyTorch implementations |
| Time-series architectures | Informer, DLinear, iTransformer | THUML Time-Series-Library adapters |
| Convolutional network | TCN | `pytorch-tcn` adapter |
| Feedforward network | MLP | `torchvision.ops.MLP` adapter |

Optuna tunes Ridge, KNN and the seven deep models against validation performance. The deep learning pipeline includes RevIN, input noise, gradient clipping, mixed precision and model-specific training settings.

<details>
<summary><strong>Notebook workflow: ten connected stages</strong></summary>

| Stage | What it produces |
| :--- | :--- |
| 1. Data loading and setup | Runtime configuration, dataset fingerprint and integrity checks |
| 2. Exploratory data analysis | Trends, distributions, ADF tests, STL, correlations and lag structure |
| 3. Preprocessing | Cyclical features, chronological splits, scaling and sliding windows |
| 4. Conventional models | Naive, Ridge and KNN benchmarks with Optuna tuning |
| 5. Deep learning | Seven model runs, checkpoints, loss curves and predictions |
| 6. Model comparison | Shared metric tables and an accuracy–latency Pareto view |
| 7. Interpretability | Ridge coefficients, SHAP and LIME |
| 8. Ablation studies | Forecast horizon, coupling mode, context length and token strategy |
| 9. Robustness | Gaussian input noise and temporal data dropout |
| 10. Model selection | Accuracy, robustness and efficiency scores with deployment recommendations |

</details>

<a id="results"></a>

## Benchmark results

**M-96 · standardized targets · ascending MAE.** Values come from [`final_metrics_m96.csv`](outputs/tables/final_metrics_m96.csv) and the notebook's saved comparison output.

| Model | MAE ↓ | RMSE ↓ | Training (s) ↓ | P95 latency (ms) ↓ | Skill score (%) ↑ |
| :--- | ---: | ---: | ---: | ---: | ---: |
| **iTransformer** | **0.353577** | **0.572823** | 54.724 | 2.618 | **46.81** |
| **DLinear** | 0.357292 | 0.581423 | 11.489 | 0.833 | 46.25 |
| MLP | 0.370468 | 0.583729 | 8.867 | 0.511 | 44.27 |
| GRU | 0.391567 | 0.599696 | 34.434 | 0.931 | 41.10 |
| LSTM | 0.399354 | 0.609593 | 33.371 | 0.696 | 39.93 |
| TCN | 0.410137 | 0.627247 | 47.477 | 1.634 | 38.31 |
| Ridge | 0.447693 | 0.661516 | 0.793 | 0.309 | 32.66 |
| Informer | 0.454480 | 0.686030 | 276.864 | 7.643 | 31.63 |
| KNN | 0.653379 | 0.872125 | 0.045 | 105.682 | 1.72 |
| Naive | 0.664785 | 1.101770 | 0.009 | 0.005 | 0.00 |

**iTransformer leads the accuracy benchmark. DLinear combines near-leading accuracy with lower training cost and latency.** Nine models meet the 50 ms P95 target; KNN records 105.682 ms.

<details>
<summary><strong>Training energy and carbon estimates</strong></summary>

The notebook calculates `EnergyKWh = TrainTimeSec / 3600 × POWER_KW` and `CO2g = EnergyKWh × 0.475 × 1000`. The saved CUDA run uses `POWER_KW = 0.25`; the CPU branch uses `0.065`.

| Model | Estimated energy (kWh) | Estimated CO₂e (g) |
| :--- | ---: | ---: |
| iTransformer | 0.003800 | 1.8051 |
| DLinear | 0.000798 | 0.3790 |
| MLP | 0.000616 | 0.2925 |
| GRU | 0.002391 | 1.1358 |
| LSTM | 0.002317 | 1.1008 |
| TCN | 0.003297 | 1.5661 |
| Ridge | 0.000055 | 0.0262 |
| Informer | 0.019227 | 9.1327 |
| KNN | 0.000003 | 0.0015 |
| Naive | 0.000001 | 0.0003 |
| **Total** | **0.032505** | **15.4400** |

</details>

### Prediction view

The notebook plots OT predictions at the first forecast step and across a complete 96-step window.

![OT prediction comparison at the first step and across a 96-step forecast window](outputs/figures/dl_prediction_comparison_ot_h1_and_96step.png)

<a id="experiments"></a>

## Ablations, robustness and explanations

### Forecast horizon

With a fixed 96-step context, iTransformer leads aggregate MAE at horizons 24, 48, 96 and 288; DLinear leads at 672 steps.

| Horizon (steps) | Duration | Ridge MAE | DLinear MAE | iTransformer MAE |
| ---: | :--- | ---: | ---: | ---: |
| 24 | 6 h | 0.339837 | 0.305010 | **0.297591** |
| 48 | 12 h | 0.409320 | 0.340102 | **0.334241** |
| 96 | 24 h | 0.448848 | 0.356503 | **0.355024** |
| 288 | 72 h | 0.502443 | 0.393802 | **0.393112** |
| 672 | 7 days | 0.586829 | **0.434631** | 0.439606 |

### Coupling, context and token strategy

| Experiment | Settings | Result |
| :--- | :--- | :--- |
| Variable coupling | `M7→M7`, `M7→OT1`, `OT1→OT1`; physical inputs | DLinear OT_MAE: **0.139144 → 0.126965 → 0.126899**. Ridge reaches **0.130333** in `OT1→OT1`. iTransformer records **0.127537** in `M7→OT1`. |
| Context length | 48 / 96 / 192 steps; forecast fixed at 96 | At 192 steps, MAE reaches **0.340169** for DLinear, **0.404294** for Ridge and **0.344283** for iTransformer. |
| Token strategy | `VToken-Original`, `VToken-GroupedLoad`, `TToken-Proxy` | Original variable tokens lead MAE at **0.354818**; grouped-load tokens lead OT_MAE at **0.138875**. The time-token proxy records MAE **0.512820**. |

Coupling experiments compare **OT_MAE**, the standardized error for the same oil-temperature target across all three modes. The token experiment uses a dedicated time-token proxy architecture and adds three auxiliary load-group channels in the grouped variant.

<details>
<summary><strong>OT error and context-length details</strong></summary>

At the 672-step horizon, DLinear and iTransformer record OT_MAE **0.217878** and **0.218181**, respectively; Ridge records **0.273883**.

| Model | MAE, context 96 | MAE, context 192 | OT_MAE, context 96 | OT_MAE, context 192 |
| :--- | ---: | ---: | ---: | ---: |
| DLinear | 0.357090 | 0.340169 | 0.138589 | 0.139024 |
| Ridge | 0.448848 | 0.404294 | 0.152506 | 0.158695 |
| iTransformer | 0.354880 | 0.344283 | 0.139131 | 0.136911 |

</details>

### Input robustness

Gaussian stress adds noise to the input tensor at standard deviations `0 / 0.01 / 0.05 / 0.10 / 0.20`. Temporal dropout zeroes all channels at randomly selected input timestamps at rates `0 / 0.05 / 0.10 / 0.20 / 0.40`. Each experiment reports changes from its own zero-perturbation baseline.

| Model | MAE increase, noise σ = 0.20 | OT_MAE increase, noise σ = 0.20 | MAE increase, dropout = 40% | OT_MAE, dropout = 40% |
| :--- | ---: | ---: | ---: | ---: |
| **DLinear** | **+2.61%** | **+12.60%** | +34.86% | **0.540554** |
| iTransformer | +2.99% | +21.42% | **+33.23%** | 0.545264 |
| Ridge | +22.55% | +48.04% | +61.99% | 0.619496 |

DLinear and iTransformer retain stronger aggregate accuracy under both perturbations than Ridge. DLinear records the lowest OT_MAE at 40% dropout and the smallest OT_MAE increase at noise σ = 0.20.

### Model explanations

Ridge's top-20 coefficient summary, grouped by base variable, places **OT** first. iTransformer's grouped SHAP ranking also places OT first, with **18.04%** of absolute attribution, while distributing contributions across physical and calendar features. SHAP explains the mean standardized prediction across target variables and forecast steps. LIME explains mean OT predictions across the forecast horizon; `OT_t-0` is the leading rule for all three models in the selected best and worst iTransformer windows.

<details>
<summary><strong>View SHAP and LIME figures</strong></summary>

![Grouped SHAP attribution for Ridge, DLinear and iTransformer](outputs/xai/xai_shap_grouped_by_variable_three_models.png)

![LIME explanations for Ridge, DLinear and iTransformer on selected best and worst windows](outputs/xai/xai_lime_local_extreme_windows_three_models.png)

</details>

<a id="artifacts"></a>

## Project structure and artifacts

```text
ETTm-Forecasting/
├── README.md
├── README.zh-CN.md
├── ettm1_end_to_end.ipynb
├── requirements.txt
├── requirements-gpu-cu128.txt
└── outputs/
    ├── configs/      # Experiment and tuning configurations
    ├── figures/      # EDA, predictions, loss curves and comparison plots
    ├── models/       # Seven PyTorch state_dict checkpoints
    ├── tables/       # Benchmark CSV exports
    └── xai/          # SHAP and LIME figures
```

Execution also creates `ETTm1.csv` in the project root and the model source checkout at `outputs/_thuml_tsl/Time-Series-Library/`.

| Artifact | Contents |
| :--- | :--- |
| [Notebook](ettm1_end_to_end.ipynb) | Executable workflow, saved outputs and all ablation/stress-test tables |
| [Global configuration](outputs/configs/config.json) | Seed, split, window, model and optimization settings |
| [Best DL configuration](outputs/configs/best_config_dl.json) | Optuna tuning results for the seven deep models |
| [M-96 benchmark](outputs/tables/final_metrics_m96.csv) | Accuracy, training time, latency, skill score and estimated energy/carbon |
| [Collected model metrics](outputs/tables/final_metrics_table_all_tasks.csv) | Core per-run metrics collected in the main comparison stage |
| [Model checkpoints](outputs/models/) | `dlinear_best.pt`, `gru_best.pt`, `informer_best.pt`, `itransformer_best.pt`, `lstm_best.pt`, `mlp_best.pt`, `tcn_best.pt` |
| [Loss curves](outputs/figures/dl_loss_curves_all_models.png) | Training and validation curves for the deep models |
| [Explanation figures](outputs/xai/) | Grouped attribution, temporal attribution and local explanations |

<a id="deployment"></a>

## Model selection and deployment guidance

The notebook's final decision combines accuracy, noise robustness, dropout robustness, inference speed and training efficiency.

The final score is the equally weighted mean of five component scores. Accuracy uses `1 − CleanMAE` from the `M7→M7` coupling experiment. Robustness uses the slopes of OT_MAE against noise and dropout levels; efficiency uses mean inference latency and training time. These four robustness/efficiency scores use min–max scaling across the three candidates, with lower slopes, latency and training time receiving higher scores.

| Role | Model | Selection basis |
| :--- | :--- | :--- |
| **Primary** | **DLinear** | Highest five-factor score: **0.813064**, combining low error, input robustness and efficient training/inference |
| **Quality fallback** | **iTransformer** | Best clean MAE among the remaining models in the final decision table; strongest main benchmark accuracy |
| **Resource fallback** | **Ridge** | Lowest inference latency among the remaining candidates and a lightweight CPU regressor |

The five-factor scores are **0.813064** for DLinear, **0.513619** for Ridge and **0.484785** for iTransformer. The quality fallback is selected by clean MAE, and the resource fallback by latency.

1. Use **DLinear** as the primary forecasting model and **OT_MAE** as the main online quality metric.
2. Select **iTransformer** when OT_MAE shows sustained degradation under normal resource availability.
3. Select **Ridge** when GPU or service resources become unavailable.
4. Trigger retraining when rolling OT_MAE increases by more than **10% for seven consecutive days**.
5. Start rollout with **5–10%** canary traffic.

These are the model-selection and operating recommendations produced by the notebook's final section.

![Five-factor model-selection radar for DLinear, Ridge and iTransformer](outputs/figures/radar_accuracy_robustness_efficiency.png)

<a id="references"></a>

## Data and references

- **Dataset:** [ETDataset](https://github.com/zhouhaoyi/ETDataset) · [ETTm1 CSV](https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTm1.csv)
- **Dataset license:** [Creative Commons Attribution-NoDerivatives 4.0 International](https://github.com/zhouhaoyi/ETDataset/blob/main/LICENSE)
- **Time-series model implementations:** [THUML Time-Series-Library](https://github.com/thuml/Time-Series-Library)
- **Libraries:** PyTorch, scikit-learn, Optuna, statsmodels, SHAP, LIME, `pytorch-tcn` and torchvision

<details>
<summary><strong>Dataset citation · BibTeX</strong></summary>

```bibtex
@inproceedings{haoyietal-informer-2021,
  author    = {Haoyi Zhou and
               Shanghang Zhang and
               Jieqi Peng and
               Shuai Zhang and
               Jianxin Li and
               Hui Xiong and
               Wancai Zhang},
  title     = {Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting},
  booktitle = {The Thirty-Fifth {AAAI} Conference on Artificial Intelligence, {AAAI} 2021, Virtual Conference},
  volume    = {35},
  number    = {12},
  pages     = {11106--11115},
  publisher = {{AAAI} Press},
  year      = {2021}
}
```

</details>

---

<div align="center">
  <a href="#top">Back to top ↑</a> · <a href="README.zh-CN.md">简体中文</a> · <a href="ettm1_end_to_end.ipynb">Explore the notebook →</a>
</div>
