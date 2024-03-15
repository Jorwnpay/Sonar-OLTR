## Sonar-OLTR

**English|[简体中文](https://github.com/Jorwnpay/Sonar-OLTR/blob/main/README_zh.md)**

This repo contains the implementation code for the ESWA 2024 article "[Open-set recognition with long-tail sonar images](https://doi.org/10.1016/j.eswa.2024.123495)". The article proposes a benchmark for Sonar Open-set Long Tail Recognition (Sonar-OLTR) and introduces a simple and effective method called "Push the right Logit Up and the wrong logit Down (PLUD)" to enhance features. This method aims to improve the model's performance in both open-set recognition and long-tail recognition tasks during training.
![plud](./img/plud.png)

## Running the Experiments

### Main Requirements

* Python == 3.6.12
* torch == 1.9.0
* torchvision == 0.10.0
* tensorboardX == 2.4.1

You can also install dependencies by

```shell
pip install -r requirements.txt
```

We recommend using anaconda to build your code environments.

### Experimental Environments

This repository is performed on an Intel Xeon E3-1275 v6 3.8 GHz central processing unit (CPU) with 32-GB RAM and an **NVIDIA GeForce RTX 2080Ti** graphic processing unit (GPU). The operating system is **Ubuntu 20.04**. The CUDA nad CUDNN version is **10.1** and **7.6.5** respectively.

### Dataset

We used the long-tailed sonar image dataset (LTSID) and the marine-debris-fls-datasets (FLSMDD) dataset to test PLUD. The FLSMDD dataset can be downloaded from [Valdenegro's repo](https://github.com/mvaldenegro/marine-debris-fls-datasets/releases/tag/watertank-v1.0). The LTSID dataset was curated from online sources and is temporarily unavailable due to copyright issues. For user convenience, the pre-prepared FLSMDD dataset can also be downloaded from this repository by cloning it and extracting all `.rar` files in the data folder. **Note:** The datasets are provided by Valdenegro, and our purpose is to facilitate users in skipping the data preparation step and directly running the code for experiments. If you plan to use these datasets in your work, please cite their article and star their Github repository to acknowledge their contributions.

```
% FLSMDD dataset is proposed in:
@inproceedings{valdenegro2021pre,
  title={Pre-trained models for sonar images},
  author={Valdenegro-Toro, Matias and Preciado-Grijalva, Alan and Wehbe, Bilal},
  booktitle={OCEANS 2021: San Diego--Porto},
  pages={1--8},
  year={2021},
  organization={IEEE}
}
```

We have released a new forward-looking sonar image dataset called NanKai Sonar Image Dataset (NKSID), which includes a total of 2617 images across 8 categories. The dataset can be downloaded from [Jorwnpay's repo](https://github.com/Jorwnpay/NK-Sonar-Image-Dataset). All the preparation work has been completed, so you just need to place the downloaded NKSID folder into the `/Sonar-OLTR/data/` directory and extract all the `.zip` files within the `NKSID` directory.

### Prepare

If you have used the original datasets from [Valdenegro's repo](https://github.com/mvaldenegro/marine-debris-fls-datasets/releases/tag/watertank-v1.0), please follow these preparation steps. However, if you have used the prepared datasets from this repository, you only need to extract all the `.rar` files in the data folder and **skip** this step.

First, download the dataset from [Valdenegro's repo](https://github.com/mvaldenegro/marine-debris-fls-datasets/releases/tag/watertank-v1.0) and organize the original files in the following structure:

```
data
├── FLSMDD
    ├── bottle
    │   ├── 1.png
    │   ├── 2.png
    │   └── ...
    ├── can
    │   ├── 1.png
    │   ├── 2.png
    │   └── ...
    └── ...
```

Secondly, run following codes to generate data direction-label list (train_abs.txt) and 10-trail 5-fold cross-validation index list (kfold_train.txt, kfold_val.txt).

```shell
# generate data direction-label list, use FLSMDD dataset as an example 
cd ./tool/
python generate_dir_lbl_list.py --dataset FLSMDD
# generate 10-trail 5-fold cross-validation index list, use FLSMDD dataset as an example 
python generate_kfold_idx_list.py --dataset FLSMDD
```

Now, you might get a data file structure like this:

```
data
├── FLSMDD
    ├── train_abs.txt
    ├── kfold_train.txt
    ├── kfold_val.txt
    ├── bottle
    │   └── ...
    ├── can
    │   └── ...
    └── ...
```

### Training

For training PLUD, here is an example for quick start,

```shell
# Demo: training on NKSID
cd ./code/
python plud.py --dataset NKSID --train_class_num 5 --test_class_num 8
```

Here are explanations of some important args,

```shell
--dataset:      "the name of dataset, can be FLSMDD or NKSID, default is NKSID"
--lr: 			"learning rate, default is 0.01"
--backbone:     "the name of backbone, default is resnet18, can be resnet18, vit_base_patch16_224.orig_in21k_ft_in1k"
--bs:			"batch size, default is 32"
--es:			"epoch size, default is 100"
--p_value:      "the trail index of 10-trail 5-fold cross-validation, default is 0"
--k_value:      "the fold index of 10-trail 5-fold cross-validation, default is 0"
--train_class_num:			"Number of class used in training, default is 5"
--test_class_num:			"Number of class used in testing, default is 8"
--includes_all_train_class:	"If required all known classes included in testing"
--method_name:	"Method Name, default is plud"
--alpha:		"alpha value for beta distribution, default is 1.5"
--gamma:		"gamma value for plud loss, default is 0.5"
--save_results: "if you want to save the validation results, default is True"
--save_models:  "if you want to save the models, default is False"
```

If you want to train PLUD via a 5-trail 5-fold cross-validation scheme, run:

```shell
# Demo: training on NKSID, using resnet18 as backbone
cd ./code/
python auto_run.py --code_dir plud.py --mode train --method_name plud
```

### Analyze Results

After training PLUD via a 5-trail 5-fold cross-validation scheme, by default, you will get y_true, and logits results in `"/output/result/{dataset}/{method}/{backbone}/"`, e.g., `"/output/result/NKSID/plud/resnet18/y_true.txt"`. Then, you can get OSFM、Macro-F1、NMA、$OSCR_{macro}$ results through:

```shell
# Demo: analyzing on NKSID, using resnet18 as backbone
cd ./code/
python analyse_result.py --dataset NKSID --method plud --backbone resnet18 --train_class_num 5 --test_class_num 8 
```

Here are explanations of some important args,

```shell
--dataset:          "the name of dataset, can be FLSMDD or NKSID, default is NKSID"
--method:           "the name of method, default is plud"
--backbone:         "the name of backbone, default is resnet18"
--thresh_mode:		"Choose a threshold mode for calculating y_hat from logit, default is tpr95"
--train_class_num:	"Number of class used in training, default is 5"
--test_class_num:	"Number of class used in testing, default is 8"
--get_osfm:         "If you want to get Macro-F1 and OSFM result, default is True."
--get_oscr:         "If you want to get OSCR result, default is True."
--get_oscr_curve:   "If you want to get OSCR curves result, default is False."
--get_nma:          "If you want to get NMA result, default is True."
--get_close_eval:   "If you want to get close-set Gmean and Macro-F1 result, default is False."
--show_pic:         "If you want to show confusion matrix or Precision-Recall curves, default is False"
```

You can also perform batch result analysis for various methods in different datasets using `auto_run.py`:

```shell
# Demo
cd ./code/
python auto_run.py --code_dir analyse_result.py --mode analyse  
```

##  Cite

If you find this code useful in your research, please consider citing us:

```
@article{jiao2024open,
  title={Open-set recognition with long-tail sonar images},
  author={Jiao, Wenpei and Zhang, Jianlei and Zhang, Chunyan},
  journal={Expert Systems with Applications},
  pages={123495},
  year={2024},
  publisher={Elsevier}
}
```

