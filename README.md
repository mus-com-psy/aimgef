# aimgef
Artificial Intelligence Music Generation Evaluation Framework

## Get Classical string quartets from KernScore
`python main.py --mode CSQ_DATA`

## Construct dataset for Transformer training
`python ./transformer/preprocessor.py`

## Train Transformer
`python model/transformer/main.py --model Transformer --style CSQ --mode TRAIN`

## Generate excerpts from trained Transformer
`python model/tranformer/main.py \
--model Transformer \
--style CSQ \
--mode TRAIN \
--src "checkpoint dir" \
--epoch 3`
