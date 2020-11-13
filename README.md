# aimgef
Artificial Intelligence Music Generation Evaluation Framework

## Construct dataset for Transformer training
`cd ./model/`

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
