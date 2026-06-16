# ImplicitStainer

Official code release for **ImplicitStainer** ([arXiv:2505.09831](https://arxiv.org/abs/2505.09831)).

ImplicitStainer performs virtual staining (e.g. H&E → IHC translation) using an
implicit neural representation. An encoder backbone (EDSR or a Swin/convolution
hybrid) extracts features from the source-domain image, and a coordinate-based
MLP decodes the target-domain stain at each pixel location. Training uses a
combination of a Smooth L1 loss and AlexNet/VGG/ResNet perceptual losses.

The code is adapted from [LIIF](https://github.com/yinboc/liif).

## Installation

```bash
pip install -r requirements.txt
```

Notes:
- The perceptual losses are imported as `generative.losses`, which is provided
  by the **MONAI GenerativeModels** package (`monai-generative`), not by core MONAI.
- A CUDA-capable GPU is required for training and inference.

## Dataset

Experiments use the **HEMIT** H&E → IHC paired dataset. Organize the data as
paired directories of source-domain (Domain A, H&E) and target-domain
(Domain B, IHC) image patches, with matching filenames across the two
directories. Configure the train/val/test paths as described below.

## Configuration

Set your data and output paths before training:

1. Edit the dataset paths in
   [`configs/train-he2ihc/train_he_to_ihc_liif.yaml`](configs/train-he2ihc/train_he_to_ihc_liif.yaml),
   **or** set them directly in the `if args.dataset == 'HEMIT'` block of the
   training script (the script overrides the config values there).
2. Set the output directory by replacing the `<Saving Directory Path>`
   placeholder (`save_dir`) in `ImplcitTrainer-Main.py`
   (and in `ImplicitTrainer-LowResolutionTraining.py` for low-resolution training).

## Training

Full-resolution training for 200 epochs with the EDSR backbone (ImplicitStainer
is selected by default inside the code):

```bash
python ImplcitTrainer-Main.py --batch_size 4 --epoch 200 --activation relu --dataPercentage full --backbone edsr --dataset HEMIT
```

Best-performing configuration (slower to train, needs larger GPUs):

```bash
python ImplcitTrainer-Main.py --batch_size 4 --epoch 200 --activation relu --dataPercentage full --backbone swin-conv-parallel-add-l --dataset HEMIT
```

Low-resolution training with high-resolution inference:

```bash
python ImplicitTrainer-LowResolutionTraining.py --batch_size 1 --epoch 1 --activation relu --dataPercentage full --backbone edsr --dataset HEMIT --lambda_p 1.0 --lambda_p1 1.0 --lambda_p2 0.0
```

### Key arguments

| Argument            | Description                                                                 | Default          |
| ------------------- | --------------------------------------------------------------------------- | ---------------- |
| `--backbone`        | Encoder: `edsr`, `swin-s`, `swin`, `swin-l`, `swin-conv-parallel-add[-small/-l]` | `edsr`     |
| `--activation`      | MLP activation: `relu`, `prelu`, `elu`                                       | `relu`           |
| `--modelType`       | MLP width/depth: `normal`, `normal-large`, `deep`                            | `normal`         |
| `--dataPercentage`  | `full` or `tenth` (train on 10% of data)                                     | `full`           |
| `--lambda_p` / `--lambda_p1` / `--lambda_p2` | Weights for AlexNet / VGG / ResNet perceptual losses       | `1.0` each       |
| `--lr`              | Learning rate                                                               | `1e-4`           |
| `--epoch`           | Number of training epochs                                                   | `200`            |
| `--resume`          | Resume from `epoch-last.pth` in the save path                               | off              |

After training, the script runs inference on the test set and reports PSNR,
SSIM, and L2.

## Pretrained models

Pretrained models will be released after paper acceptance.

## Citation

If you use this code, please cite:

```bibtex
@article{kataria2025implicitstainer,
  title={ImplicitStainer},
  author={Kataria, Tushar and Beatrice Knudsen, Shireen Y. Elhabian},
  journal={arXiv preprint arXiv:2505.09831},
  year={2025}
}
```

## Acknowledgements

This code is adapted from [LIIF](https://github.com/yinboc/liif).

## License

Released under the BSD 3-Clause License. See [LICENSE](LICENSE).
