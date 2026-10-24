# References

Factual list of the software, pre-trained weights and methods used in this project. Keep it in sync with the
runs that actually end up in the report (remove entries for presets that were never trained).

## Software

- PyTorch 2.14 and torchvision 0.29 — Paszke et al. (2019). *PyTorch: An imperative style, high-performance deep learning library.* NeurIPS.
- timm 1.0 (PyTorch Image Models) — Wightman, R. (2019). https://github.com/huggingface/pytorch-image-models
- scikit-learn 1.9 — Pedregosa et al. (2011). *Scikit-learn: Machine learning in Python.* JMLR 12.
- NumPy, pandas, SciPy, matplotlib, seaborn, Pillow, OpenCV, papermill (versions in `requirements.txt`).

## Pre-trained weights (downloaded through timm from the Hugging Face Hub)

| used in | timm name | architecture paper | pre-training |
|---|---|---|---|
| notebook 02 (embeddings), `abl_*`, `effnet_b0_shift` | `efficientnet_b0.ra_in1k` | Tan & Le (2019), *EfficientNet: Rethinking model scaling for CNNs*, ICML | ImageNet-1k, RandAugment recipe in timm |
| `convnext_tiny_*` | `convnext_tiny.fb_in22k_ft_in1k` | Liu et al. (2022), *A ConvNet for the 2020s*, CVPR | ImageNet-22k, fine-tuned on ImageNet-1k |
| `vit_small_*` | `vit_small_patch16_224.augreg_in21k_ft_in1k` | Dosovitskiy et al. (2021), *An image is worth 16x16 words*, ICLR; Steiner et al. (2022), *How to train your ViT?*, TMLR | ImageNet-21k (AugReg), fine-tuned on ImageNet-1k |

ImageNet: Deng et al. (2009), *ImageNet: A large-scale hierarchical image database*, CVPR.

## Methods

- AdamW — Loshchilov & Hutter (2019), *Decoupled weight decay regularization*, ICLR.
- Cosine learning-rate schedule — Loshchilov & Hutter (2017), *SGDR: Stochastic gradient descent with warm restarts*, ICLR.
- Label smoothing — Szegedy et al. (2016), *Rethinking the Inception architecture for computer vision*, CVPR.
- Weight averaging (EMA) — Polyak & Juditsky (1992), *Acceleration of stochastic approximation by averaging*, SIAM J. Control Optim.
- Batch normalisation — Ioffe & Szegedy (2015), ICML. Stochastic depth — Huang et al. (2016), ECCV.
- Mixed precision training — Micikevicius et al. (2018), ICLR.
- Difference hash (dHash) for near-duplicate detection — Krawetz, N. (2013), *Kind of like that*, The Hacker Factor Blog.
- Adversarial validation (train-vs-test classifier) — a common Kaggle diagnostic; see e.g. Pan et al. (2020),
  *Adversarial validation approach to concept drift problem in user targeting automation systems at Uber*, arXiv:2004.03045.
