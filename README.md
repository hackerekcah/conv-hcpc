# Paper title
Capturing Temporal Dependencies Through Future Prediction for CNN-based Audio Classifiers

# Requirements
* See requirements.yml

# Repository structure
* `config`
  * Experiment configurations and Hyper-parameters for different datasets.
* `cpc`
  * Contrastive Predictive Coding loss module `cpc/cpc.py`
  * CPC accuracy for different prediction steps `cpc/metric.py`
* `data`
  * Datasets classes `esc_dataset.py`, `gtzan.py`, `urbansound8k.py`.
  * Augmentation methods such as Pitch Shifting. `data_transformer.py`
* `layers`
  * Input log melspectrogram on-gpu calculation.
  * Conv1d blocks, causal modules, fc layers etc.
  * Loss functions with label smoothing
* `models`
  * The backbone CNN model with auxiliary prediction tasks.
* `utils`
  * Saving checkpoint, keeping tracking of training histories.
* `engine_cpc.py`
  * Training and evaluation methods
* `main_cpc.py`
  * data loading
  * model instantiation
  * train and evaluation
  * tracking lr, acc, loss with Tensorboard.
  * model saving
# Datasets
* Under `data` sub-folder
* Need to point `root` parameter of each dataset class to your own path of the dataset.

# Training & Evaluation
``` bash
python main_cpc.py --cfg <cfg_file>

# Reproducing the HCPC model on ESC-50 Datasets
python main_cpc.py --cfg config/esc_conv1d_hcpc_folds_config.yaml

# Reproducing the HCPC model on GTZAN Datasets
python main_cpc.py --cfg config/gtzan_conv1d_hcpc_folds_config.yaml

# Reproducing the HCPC model on UrbanSound8K Datasets
python main_cpc.py --cfg config/urban_conv1d_hcpc_folds_config.yaml

```

