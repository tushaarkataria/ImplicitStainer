
## Code Release for ImplicitStainer(https://arxiv.org/abs/2505.09831)

# How to Used the Repo to Train

1. Edit "https://github.com/tushaarkataria/ImplicitStainer/blob/main/configs/train-he2ihc/train_he_to_ihc_liif.yaml" to add training, validation and test paths.

2. Edit "ImplcitTrainer-Main.py" to add save_dir paths and other datasets you want to run on.

3. Training command

"python ImplcitTrainer-Main.py --batch_size 1 --epoch 200 --activation relu --dataPercentage full --backbone edsr --dataset HEMIT "

# Best model but slow in training and required larger GPUs.
"python ImplcitTrainer-Main.py --batch_size 1 --epoch 200 --activation relu --dataPercentage full --backbone swin-conv-parallel-add-l --dataset HEMIT " 


Code is Adapted from "https://github.com/yinboc/liif"
