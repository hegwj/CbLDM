# CbLDM
This is a code project about Condition-based Latent Diffusion Model, by using the deep learning method can learn the structure of some particles from the noise. It can recover the structure according to a specific function Pair Distribution Function.
1. [CbLDM](#CbLDM)
2. [Train model](#Train-model)
3. [Predict](#Predict)
4. [Notes on the file](#Notes-on-the-file)
5. [Authors](#Authors)
# Train model
To train your own CbLDM model simply run:
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

[gif](gif): An example was deposited showing the HCP structure being predicted by CbLDM constant diffusion.

[modules](modules): It includes the model architecture file for CbLDM and dataloader.py.

[utils](utils): It contains the python packages that will be used in CbLDM.

# Authors
__Jiarui Cao__<sup>1</sup> 
__Zhiyang Zhang__<sup>1</sup> 
__Heming Wang__<sup>1</sup> 
__Ran Gu__<sup>1</sup> 

<sup>1</sup> School of Statistics and Data Science, NanKai University, Tianjin, Weijin Road 94, China.

If you have any questions, need improvements, or have bugs, please contact us via GitHub or email: __tjzzyang@163.com__
