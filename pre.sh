#!/bin/sh

python preprocess.py --dataset NYC &
python preprocess.py --dataset TKY &
python preprocess.py --dataset CA &

wait

python build_graph.py --dataset NYC &
python build_graph.py --dataset TKY &
python build_graph.py --dataset CA &

wait

python ui_graph.py --dataset NYC &
python ui_graph.py --dataset TKY &
python ui_graph.py --dataset CA &

wait

echo "Preprocessing completed."
