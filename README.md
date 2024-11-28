# CbLDM
This is a code project about Condition-based Latent Diffusion Model, by using the deep learning method can learn the structure of some particles from the noise. It can recover the structure according to a specific function Pair Distribution Function.
1. [CbLDM](#CbLDM)

## Install requirements
See the [install](/install) folder. 

## Simulate data
See the [data](/data) folder. 

## Train model
To train your own DeepStruc model simply run:
```
python train.py
```
Before training, you need to replace the paths in train.py such as “model_path”, “data_path” with your own.
