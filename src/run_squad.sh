#!/bin/bash
source /w/340/lucbr/miniconda3/bin/activate
export PYTHONPATH="$PYTHONPATH:/w/340/lucbr/gramba/src"
conda activate gramba
python3 -u src/train/train_squad.py
exit 