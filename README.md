# Lung CT Classification (IQ-OTH/NCCD)

Fine-tunes an ImageNet-pretrained **ResNet-50** (PyTorch) to classify chest CT slices into **Benign**, **Malignant** or **Normal**, using an augmented version of the IQ-OTH/NCCD lung cancer dataset.

> **Educational project, not a medical device.** The model has not been clinically validated and must not be used for diagnosis or any medical decision.

All code is in [`Lung_Cancer_Detection.ipynb`](Lung_Cancer_Detection.ipynb).

## Dataset

The notebook uses an augmented version of the **IQ-OTH/NCCD lung cancer dataset** (CT slices collected in two Iraqi hospitals in 2019). It is not included in this repository because of its size and because it has its own terms of use. Get the original from [Mendeley Data](https://data.mendeley.com/datasets/bhmdr45bh2) and check its terms before using or sharing it.

The augmented version used here has this layout (place the folder next to the notebook):

```
Augmented IQ-OTHNCCD lung cancer dataset/
├── Benign cases/
├── Malignant cases/
├── Normal cases/
└── Augmentation Details.txt
```

| Class     | Original slices | After augmentation |
|-----------|----------------:|-------------------:|
| Benign    | 120             | 1,200              |
| Malignant | 561             | 1,201              |
| Normal    | 416             | 1,208              |
| **Total** | **1,097**       | **3,609**          |

About 70% of the files (2,512 of 3,609) are augmented copies of the 1,097 originals: flips, rotation, color jitter, contour crop, Gaussian blur, sharpening, contrast, and histogram equalization.

### Citation

If you use this dataset, please cite it as requested by its authors:

- Alyasriy, H.; Al-Huseiny, M. (2023), *The IQ-OTH/NCCD lung cancer dataset*, Mendeley Data, V4, doi: [10.17632/bhmdr45bh2.4](https://doi.org/10.17632/bhmdr45bh2.4)
- Al-Yasriy, H. F., Al-Husieny, M. S., Mohsen, F. Y., Khalil, E. A., Hassan, Z. S. (2020), *Diagnosis of Lung Cancer Based on CT Scans Using CNN*, IOP Conference Series: Materials Science and Engineering, vol. 928.

## Why the notebook uses only the original slices

If you split the whole augmented folder at random, copies of the same CT slice end up in both the training and validation sets. The model is then partly tested on images it has effectively already seen, and the accuracy looks much better than it would on new scans.

So by default (`ORIGINALS_ONLY = True`) the notebook:

1. Keeps only the 1,097 original slices.
2. Splits them 70/15/15 into train/validation/test, stratified by class, with a fixed seed.
3. Applies light augmentation (flip, small rotation) on the fly to the training set only.
4. Uses a class-weighted loss, since the originals are imbalanced (120 benign vs 561 malignant).
5. Picks the best checkpoint by validation loss, then reports a per-class classification report and a confusion matrix on the untouched test split.

Set `ORIGINALS_ONLY = False` to train on every file. That reproduces the leaky setup, so the scores it prints will be inflated.

## Setup

Python 3.10+ is recommended. A CUDA GPU makes training much faster, but the notebook also runs on CPU.

```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook Lung_Cancer_Detection.ipynb
```

Put the dataset folder next to the notebook, then run all cells. Training writes the best weights to `best_model.pt`, which is git-ignored.

## Results

The first version of this project split the full augmented folder 80/20 at random and reached a peak validation accuracy of 98.2% (epoch 5 of 10), ending at 93.8%. Because of the leakage described above, that is **not** a measure of how well the model works on unseen scans.

Running the notebook as it is now prints the per-class precision, recall and F1, plus a confusion matrix, on the held-out test split.

## Limitations

- **Patient-level leakage remains.** The original dataset has 1,190 slices from only 110 patients (40 malignant, 15 benign, 55 normal). This version has no patient IDs, so slices from the same patient can still fall into different splits, and test scores are probably still optimistic. A proper evaluation would split by patient.
- **Few benign cases.** Only 15 benign patients (120 slices) exist, so benign metrics are very noisy.
- **Single source.** All scans come from two hospitals in one region, using one scanner model, so the results may not carry over to other populations or equipment.
- **2D slices only**, with no use of the 3D scan context.
