[README (1).md](https://github.com/user-attachments/files/31869766/README.1.md)

# Chest X-Ray Classification — VGG19 + Fine-Tuning

Multi-class chest X-ray classifier (Bacterial, Covid, Enlarged Cardiomediastinum, Normal, Viral) built with a VGG19 backbone: frozen transfer learning followed by fine-tuning of the last conv block.

## Data

- 5 classes, 250 train images / 150 test images
- [Dataset](https://drive.google.com/drive/folders/1KVggIWvnG4tmZ9MSkLBAn5XvO4GMcu3z?usp=drive_link)
- Expected layout:
  ```
  Data/
    train/<class_name>/*.png
    test/<class_name>/*.png
  ```

## Method

- **Backbone:** VGG19 (ImageNet weights), frozen
- **Head:** GlobalAveragePooling → BatchNorm → Dropout → Dense(256, L2) → Dropout → Dense(5, softmax)
- **Training:** frozen backbone first, then fine-tuned by unfreezing `block5_conv4` at a lower learning rate
- **Augmentation:** flip, rotation, zoom, contrast
- **Split:** stratified train/val split (train_test_split), held-out test set

## Setup

```bash
pip install tensorflow scikit-learn matplotlib seaborn
```

Download the [VGG19 no-top weights](https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5) and set `DATA_DIR` / `WEIGHTS_PATH` at the top of the notebook.

## Run

Open `chest_xray_vgg19.ipynb` and run top to bottom.

## Results

| Metric | Value |
|---|---|
| Test accuracy | — |
| Test loss | — |

## Notes

- "Enlarged Cardiomediastinum" is a radiological finding, not an infection type — worth keeping in mind alongside the other 4 classes.
- Small dataset (250 train images) — results should be read with that in mind.
