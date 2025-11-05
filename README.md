# Variational Autoencoder (VAE) on MNIST — Full Analysis & Visualization

A PyTorch implementation of a Variational Autoencoder (VAE) trained on the MNIST dataset.
This repository trains a 2-D latent VAE, inspects latent representations, trains a simple classifier on the latent space, computes reconstruction quality metrics (SSIM / PSNR / MSE), and produces a set of publication-quality visualizations (loss curves, latent scatter, reconstructions, generated samples, interpolation, per-digit latent KDEs, ROC / confusion matrix, etc.).

---

## Features

* Train a fully-connected VAE (encoder + decoder).
* 2D latent space for easy visualization.
* Reconstruction + KL loss tracking and plots.
* Save generated samples, interpolation, per-digit latent visualizations, and overall summary figure.
* Compute reconstruction quality metrics: SSIM, PSNR, MSE.
* Train a RandomForest classifier on the learned latent vectors and compute classification metrics and ROC-AUC.
* Robust plotting (fixed subplot indexing bug that caused `ValueError: num must be an integer with 1 <= num <= ...`).
* Saves model weights (`vae_mnist_model.pth`) and several PNG visualizations.

---

## Files produced by the code

* `vae_mnist_model.pth` — trained model weights
* `generated_samples.png` — 5×5 grid of random samples decoded from latent space
* `interpolation.png` — interpolation sequence between two latent vectors
* `vae_mnist_complete_analysis.png` — main multi-panel summary figure (losses, latent scatter, sample reconstructions, confusion matrix, ROC)
* `latent_space_per_digit.png` — per-digit latent KDE/scatter plots
* `ssim_distribution.png`, `psnr_distribution.png` — reconstruction quality histograms
* `Handwritten.ipynb` — the Jupyter notebook (if included / provided)

---

## Quick Start

### Requirements

Recommended: a machine with CUDA if you plan to train faster. The script auto-detects device:

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

Install dependencies (pip):

```bash
pip install torch torchvision numpy matplotlib seaborn scikit-learn scikit-image scipy
```

Or create a conda environment:

```bash
conda create -n vae-mnist python=3.10
conda activate vae-mnist
pip install torch torchvision numpy matplotlib seaborn scikit-learn scikit-image scipy
```

> Tested with PyTorch 2.x and common versions of the other libraries. If you use an older/newer environment, minor API differences may appear.

### Run the notebook / script

If you have the `Handwritten.ipynb` notebook:

```bash
jupyter lab Handwritten.ipynb
# or
jupyter notebook Handwritten.ipynb
```

If you saved the notebook as a script (e.g. `train_vae.py`), run:

```bash
python train_vae.py
```

During training, the script prints epoch-by-epoch progress and saves visualizations at the end.

---

## How the code is organized

* **Model**: `VAE` (encoder → `fc_mu`, `fc_logvar`; reparameterize; decoder).
* **Loss**: BCE reconstruction (sum) + KLD term.
* **Training loop**: standard PyTorch training / validation loop with logged train/test losses.
* **Evaluation**:

  * Extract mean (`mu`) per example as latent representation.
  * Train a RandomForest classifier on latent vectors and report accuracy, precision, recall, F1.
  * Compute ROC-AUC per class and micro-average.
* **Reconstruction metrics**: SSIM, PSNR, MSE computed with correct image shape handling (supports `(N,1,28,28)` or `(N,784)`) to avoid shape errors.
* **Visualizations**: arranged into multiple saved PNGs. Subplot index bug that produced invalid subplot numbers has been fixed by placing interpolations and generated samples in separate figures and ensuring valid subplot indices.

---

## Usage examples

Typical workflow (in notebook or script):

1. Train the model (runs for `EPOCHS` epochs).
2. Extract latent vectors from the test set.
3. Train a RandomForest classifier on the latent vectors (`sklearn`).
4. Compute SSIM/PSNR/MSE for `sample_size` reconstructed images.
5. Save visualizations.

Key configuration variables at the top of the script/notebook:

```python
BATCH_SIZE = 128
LATENT_DIM = 2          # keep 2 for visualization
HIDDEN_DIM = 400
LEARNING_RATE = 1e-3
EPOCHS = 20
INPUT_DIM = 784
sample_size = 1000      # for computing SSIM/PSNR/MSE
```

To change training length or latent dim, modify `EPOCHS` and `LATENT_DIM` respectively.

---

## Notes & Troubleshooting

* **ValueError: subplot num must be an integer...** — this was caused by invalid subplot indexing like `plt.subplot(2, n_steps, i + 1 + 100)` (which produced indices beyond the grid size). The code in this repo separates interpolation and generated samples into their own figures and uses safe indexing (e.g. `plt.subplot(1, n_steps, i+1)`).
* If you encounter shape errors in SSIM/PSNR calculation, ensure the arrays passed to the metric functions are 2D images of shape `(28, 28)` and values are in `[0, 1]`. The repo contains a helper `compute_image_metrics` which normalizes and reshapes inputs robustly.
* For reproducibility, set seeds (optional) in the notebook/script:

  ```python
  import random, numpy as np, torch
  seed = 42
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
  ```

---

## Results (example)

After training for `EPOCHS=20` on MNIST:

* The script prints classification metrics on latent space (Accuracy, Precision, Recall, F1).
* It prints reconstruction metrics (SSIM, PSNR, MSE).
* It saves a set of PNGs listed above for quick inspection and for including in reports / README.

(Exact numbers will depend on random initialization and training hyperparameters — check the notebook output for the most recent run.)

---

## Extending the project

* Replace fully-connected encoder/decoder with convolutional architectures (ConvVAE) for better image fidelity.
* Increase `LATENT_DIM` for better generative capacity; use dimensionality reduction (t-SNE / UMAP) for visualization.
* Replace the RandomForest classifier with a logistic regression or an MLP trained end-to-end on latent vectors.
* Add conditional VAE (CVAE) to condition generation on digit label.
* Export generated images or create a simple Flask / Streamlit app to demonstrate generation interactively.

---

## License

This project is released under the **MIT License** — see `LICENSE` for details.

---

## Citation / Attribution

If you use this code in academic work, please cite the project in an appropriate manner and consider referencing standard VAE literature (e.g., Kingma & Welling, 2014).

---

## Contact / Contributions

Created and maintained by the author. Contributions are welcome — open issues and pull requests on GitHub.
If you want help adapting the notebook into a modular script, Docker image, or Colab-ready notebook, feel free to open an issue or a PR.

---

## Example `requirements.txt`

```
torch
torchvision
numpy
matplotlib
seaborn
scikit-learn
scikit-image
scipy
```
