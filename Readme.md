
## Code Release for ImplicitStainer(https://arxiv.org/abs/2505.09831)

# How to Used the Repo to Train

1. Edit "https://github.com/tushaarkataria/ImplicitStainer/blob/main/configs/train-he2ihc/train_he_to_ihc_liif.yaml" to add training, validation and test paths.

2. Edit "ImplcitTrainer-Main.py" to add save_dir paths and other datasets you want to run on.

3. Training command, runs for 200 epochs on full data, with EDSR backbone. Implicit Stainer is selected inside the code by default

``` python ImplcitTrainer-Main.py --batch_size 4 --epoch 200 --activation relu --dataPercentage full --backbone edsr --dataset HEMIT ```

# Best model but slow in training and required larger GPUs.
``` python ImplcitTrainer-Main.py --batch_size 4 --epoch 200 --activation relu --dataPercentage full --backbone swin-conv-parallel-add-l --dataset HEMIT ```

4. Low resolution training Command:

``` python ImplicitTrainer-LowResolutionTraining.py --batch_size 1 --epoch 1 --activation relu --dataPercentage full --backbone edsr --dataset HEMIT --lambda_p 1.0 --lambda_p1 1.0 --lambda_p2 0.0 ```


## Pretrained models to be released after Acceptance.

Code is Adapted from "https://github.com/yinboc/liif"
