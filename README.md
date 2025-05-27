# GIRAM

Source code for **"Model-Agnostic Continual Learning for Next POI Recommendation"**.

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
