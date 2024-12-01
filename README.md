# CbLDM
This is a code project about Condition-based Latent Diffusion Model, by using the deep learning method can learn the structure of some particles from the noise. It can recover the structure according to a specific function Pair Distribution Function.
1. [CbLDM](#CbLDM)
2. 
# Train model
To train your own DeepStruc model simply run:
```
python train.py
```
Before training, you need to replace the paths in train.py such as “model_path”, “data_path” with your own.

# Predict
To predict a coordinate of an atomic structure using CbLDM on a PDF:
```
python predict.py
```

# Notes on the file
Instructions for using the files involved in this project are given below:

[gif](\gif): An example was deposited showing the HCP structure being predicted by CbLDM constant diffusion.
