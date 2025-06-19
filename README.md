# GIRAM

Source code for **"Effcient Model-Agnostic Continual Learning for Next POI Recommendation"**.

Datasets and derived POI categories are available in the `./data` directory.

### Requirements

Make sure the following Python packages are installed:

- `torch`
- `torch_geometric`
- `torchsde`
- `networkx`
- `pandas`
- `numpy`

### Preprocessing
To prepare the datasets, run the preprocessing script:
```
sh pre.sh
```

### Pretraining
To train the base model on the initial data block (datasets: NYC, TKY, CA), run:
```
python main_Flashback.py --mode pretrain --dataset NYC
```

### Continual Updating
To perform continual learning with different update strategies, run:
```
python main_Flashback.py --mode memory --dataset NYC
```
Available modes:
- `memory`: Applies the proposed GIRAM method for continual learning.
- `finetune`: Incrementally fine-tunes the model using only new data.
- `retrain`: Retrains the model from scratch using all accumulated data.

### Acknowledgement
Our code is based on the following works:
- Flashback: https://github.com/eXascaleInfolab/Flashback_code
- GETNext: https://github.com/songyangme/GETNext
- DiffPOI: https://github.com/Yifang-Qin/Diff-POI
