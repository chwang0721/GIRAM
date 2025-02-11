# GIRAM
Code for “Model-Agnostic Continual Learning for Next POI Recommendation"
### Requirements
- torch
- torch_geometric
- torchsde
- networkx
### Preprocessing
Run
``
sh pre.sh
``
### Pretraining
```
python main_Flashback.py --mode pretrain --dataset NYC
```
### Continual Updating
```
python main_Flashback.py --mode memory --dataset NYC
```
Here, `mode` is the method for continual learning, `memory` denotes the GIRAM model, `finetune` and `retrain` are baselines.
