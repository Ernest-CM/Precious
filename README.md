# Tomato Leaf Disease Detector — Chapter 4 Implementation

**Author:** Idowu Oluwasemilore Precious (22CD009369 / 2201385)
**Department:** Computer Science, Landmark University
**Date:** January, 2026

This implementation realises the design defined in `revised.md`:
a lightweight MobileNetV2 transfer-learning classifier for 10 tomato leaf
classes, evaluated on both a controlled and a field-style test split,
and delivered as a local Streamlit web app.

The walkthrough below uses **Anaconda Prompt** on Windows.

---

## 1. Folder layout

```
Precious/
├── app/
│   └── ui.py                  # Streamlit interface
├── scripts/
│   ├── download_dataset.py    # Kaggle PlantVillage downloader
│   ├── prepare_field_split.py # Synthesises a field-style test set
│   └── smoke_test.py          # Tiny end-to-end pipeline check
├── src/
│   ├── settings.py            # Paths + config loader
│   ├── dataset.py             # Builds the train/val/test index
│   ├── augment.py             # Training and field-distortion augmentation
│   ├── pipeline.py            # tf.data pipelines + class weights
│   ├── model_builder.py       # MobileNetV2 transfer-learning model
│   ├── train.py               # Two-stage training entrypoint
│   ├── evaluate.py            # Test evaluation + confusion matrices
│   ├── infer.py               # Single-image inference (used by UI + CLI)
│   └── advisories.py          # Plain-language disease advice
├── config.yaml                # Hyper-parameters, class list, paths
├── environment.yml            # Conda environment spec
├── requirements.txt           # Pip fallback
├── project_document.txt       # Main write-up (Chs 1-3)
├── revised.md                 # Alignment revisions
└── README.md                  # This file
```

Generated at runtime (gitignored):

```
data/raw/PlantVillage/<class>/...
data/field/<class>/...            (real or simulated)
data/prepared/index.csv           (split assignments)
models/tomato_mobilenetv2_*.keras
models/tomato_mobilenetv2.tflite
reports/training_summary.json
reports/evaluation.json
reports/confusion_controlled.png
reports/confusion_field.png
reports/classification_report_*.txt
```

---

## 2. One-time setup (Anaconda Prompt)

Open **Anaconda Prompt** and run from this folder:

```bat
conda env create -f environment.yml
conda activate tomato-leaf
```

If that environment already exists, update it with:

```bat
conda env update -f environment.yml --prune
```

Verify the install:

```bat
python -c "import tensorflow as tf; print('TF', tf.__version__)"
```

---

## 3. Get the dataset

You need a Kaggle account with an API token saved to
`%USERPROFILE%\.kaggle\kaggle.json` (see the
[Kaggle API docs](https://www.kaggle.com/docs/api)).

Then, in the Anaconda Prompt:

```bat
python scripts\download_dataset.py
```

This downloads the public PlantVillage dataset, copies the **10 tomato
classes** into `data\raw\PlantVillage\`, and tells you the per-class count.

If you already have the dataset on disk, skip the script and just lay
files out as `data\raw\PlantVillage\<class_name>\*.jpg`, using the class
folder names listed in `config.yaml`.

### Optional: field-style test set

To honour the `revised.md` §3.20 promise of a generalisation-aware
evaluation, populate `data\field\<class>\` with one of the following:

- **Recommended:** drop in real field-captured tomato leaf photos.
- **Fallback:** synthesise them with heavy distortion (different
  lighting, blur, occlusion, noise):

  ```bat
  python scripts\prepare_field_split.py --per-class 50
  ```

The evaluation report will tag the result as `real` or `simulated`
honestly, so the limitation is disclosed in Chapter 4.

---

## 4. Build the train/val/test index

```bat
python -m src.dataset
```

This writes `data\prepared\index.csv` with one row per image and the
following splits:

| Split | Source | Purpose |
|---|---|---|
| `train` (70 %) | PlantVillage | Fit the model |
| `val` (15 %) | PlantVillage | Tune hyper-parameters, early stopping |
| `test_controlled` (10 %) | PlantVillage held-out | Headline accuracy |
| `test_field` or `test_field_simulated` (5 %) | `data/field/` or distorted slice | Generalisation accuracy |

---

## 5. Smoke test (optional but recommended)

Confirms the wiring without needing the full dataset:

```bat
python scripts\smoke_test.py
```

It builds three classes of dummy images, runs one batch through training,
and prints `[ok] smoke test passed`. After this, rebuild the real index:

```bat
python -m src.dataset
```

---

## 6. Train the model

```bat
python -m src.train
```

What happens:

1. **Stage 1** — only the classifier head is trained (MobileNetV2 frozen).
   ~15 epochs, learning rate 1e-3.
2. **Stage 2** — the top portion of MobileNetV2 is unfrozen and fine-tuned.
   ~10 epochs, learning rate 1e-5.
3. Best weights from each stage are saved as Keras files in `models/`.
4. The final model is also exported to **TensorFlow Lite** at
   `models\tomato_mobilenetv2.tflite` for future mobile use.
5. `reports\training_summary.json` records timing and best validation accuracy.

Training assumes Adam optimizer, categorical cross-entropy loss, dropout
0.4, and class-weighting to handle imbalance (see `config.yaml`).

To shorten training while debugging, edit `config.yaml`:

```yaml
training:
  epochs_head: 2
  epochs_finetune: 1
```

---

## 7. Evaluate

```bat
python -m src.evaluate
```

Outputs in `reports/`:

- `evaluation.json` — accuracy, macro / weighted precision, recall, F1
  for the **controlled** split and the **field-style** split.
- `confusion_controlled.png` and `confusion_field.png`.
- `classification_report_controlled.txt` and `classification_report_field.txt`
  with per-class precision/recall/F1.

Quote the controlled-vs-field gap in Chapter 4 as the generalisation
finding promised by `revised.md`.

---

## 8. Run the web app

```bat
streamlit run app\ui.py
```

Streamlit prints a local URL such as `http://localhost:8501`. Open it in a
browser, upload (or capture) a tomato leaf photo, and the app shows:

- Predicted class and confidence.
- A short advisory (cause, symptoms, recommended action) for that disease.
- Top-3 candidate classes with confidence scores.

The app refuses to start if no model has been trained yet, with a clear
error message telling you to run `python -m src.train` first.

---

## 9. Predict from the command line

```bat
python -m src.infer path\to\leaf.jpg
```

Example output:

```
Predicted: Early Blight (96.42% confidence)
Top candidates:
  - Early Blight: 96.42%
  - Septoria Leaf Spot: 2.31%
  - Target Spot: 0.84%
```

---

## 10. Suggested Chapter 4 narrative outline

Chapter 4 of the project document should walk through:

1. **System overview** — the architecture diagram from `revised.md` §3.18.
2. **Implementation environment** — Python 3.10, TensorFlow 2.15, Keras,
   OpenCV, Streamlit, Anaconda on Windows.
3. **Dataset and split** — counts produced in §4 above.
4. **Model architecture** — MobileNetV2 + GAP + BatchNorm + Dense(256) +
   Dropout(0.4) + Softmax(10).
5. **Training procedure** — two-stage (head, then fine-tune); curves
   from `reports\history_head.csv` and `reports\history_finetune.csv`.
6. **Results** — controlled accuracy, field-style accuracy, confusion
   matrices, per-class metrics, top mis-classifications.
7. **Deployment** — screenshots of the Streamlit app classifying three
   leaves (one healthy, two diseased).
8. **Limitations and future work** — disclose whether the field split is
   `real` or `simulated`, and note the TFLite export as the route to a
   future mobile front-end.

---

## 11. Quick command summary

```bat
conda env create -f environment.yml
conda activate tomato-leaf

python scripts\download_dataset.py
python scripts\prepare_field_split.py --per-class 50   :: optional
python -m src.dataset

python scripts\smoke_test.py                           :: optional

python -m src.train
python -m src.evaluate

streamlit run app\ui.py
```
