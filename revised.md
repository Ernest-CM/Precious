# REVISED SECTIONS — TOMATO PLANT DISEASE DETECTION PROJECT

**Author:** Idowu Oluwasemilore Precious (22CD009369 / 2201385)
**Department:** Computer Science, Landmark University
**Date:** January, 2026

---

## Purpose of This Revision

This document supplements the main project write-up (`project_document.txt`) by resolving alignment gaps identified during a structural review of the project. The original chapters establish a strong foundation, but four issues were flagged:

1. **Scope contradicts real-world variability requirement** *(critical)*
2. **Real-time automated framework not defined in methodology** *(critical)*
3. **Accessibility / affordable deployment not addressed** *(warning)*
4. **Dataset strategy and class definition unclear** *(warning)*

The revisions below rewrite or extend the affected sub-sections so that the **Problem Statement**, **Aim & Objectives**, **Scope**, and **Methodology** become internally consistent and Chapter 4 (Implementation) can be delivered without overreach.

---

## 1. Revised Section 1.2 — Statement of Problem (Refined)

Tomato production plays a vital role in food security, nutritional balance, and the economic sustainability of farmers worldwide. Despite its importance, tomato cultivation continues to suffer significant yield losses caused by foliar diseases such as early blight, late blight, bacterial spot, leaf mold, septoria leaf spot, target spot, mosaic virus, and yellow leaf curl virus. In developing regions, the problem is intensified by limited access to extension officers and diagnostic laboratories.

The dominant method of detection — visual inspection by farmers — is subjective, experience-dependent, and unreliable because many tomato diseases present visually similar early-stage symptoms. Misdiagnosis leads to inappropriate pesticide use, increased production costs, environmental damage, and continued disease spread.

Although deep-learning-based plant disease classifiers have been widely reported in the literature, three specific gaps remain unresolved and form the focus of this study:

1. **Generalization gap** — Most published models are trained and evaluated on the PlantVillage dataset, where leaves are imaged against uniform backgrounds under controlled lighting. Reported accuracies above 98 % drop sharply on field-captured images that contain occlusion, shadow, soil, and mixed foliage.
2. **Deployment gap** — Many high-accuracy architectures (e.g., DenseNet, ViT, ResNet-152) are too large to run on mobile or low-power devices, which are the realistic delivery platforms for smallholder farmers.
3. **Usability gap** — Research prototypes rarely include a complete inference pipeline (image capture → preprocessing → classification → result presentation) that a non-technical farmer can use.

The core problem addressed in this study is therefore the absence of an automated tomato leaf disease detection system that is **(a)** accurate on both controlled and moderately variable field-style images, **(b)** lightweight enough to run on a commodity laptop or mid-range smartphone, and **(c)** delivered as a working, end-to-end inference application rather than a notebook experiment.

---

## 2. Revised Section 1.3 — Aim and Objectives (Refined)

### Aim

To design, implement, and evaluate a lightweight Convolutional Neural Network (CNN) — built using transfer learning — that automatically classifies common tomato leaf diseases from images and is delivered as a working inference application capable of running on a standard laptop or mid-range smartphone, without requiring an internet connection or expert diagnostician.

### Specific Objectives

1. To **acquire and curate** a labeled tomato leaf image dataset combining the public PlantVillage collection with a smaller set of field-style images (varied backgrounds, lighting, and orientations) covering **ten target classes** (nine diseases plus healthy).
2. To **preprocess** the dataset using resizing, normalization, and a defined augmentation pipeline (rotation, flipping, brightness/contrast jitter, random cropping) that simulates real-world variability.
3. To **design and train** a CNN classifier using transfer learning from a mobile-friendly backbone (MobileNetV2 or EfficientNet-B0) optimized for accuracy-vs-size trade-off.
4. To **evaluate** the trained model using accuracy, precision, recall, F1-score, per-class confusion matrices, and a held-out field-style test split that measures generalization beyond PlantVillage.
5. To **deploy** the trained model as an end-to-end inference application (a local web interface built with Streamlit or Flask) that accepts a leaf image and returns the predicted disease class with a confidence score.

---

## 3. Revised Section 1.6 — Scope of Study (Refined)

This study is bounded as follows:

| Aspect | In Scope | Out of Scope |
|---|---|---|
| Plant part | Tomato **leaf** images | Fruits, stems, roots, whole-plant imagery |
| Conditions | PlantVillage (controlled) **and** a curated mixed-condition test split simulating field variability (lighting jitter, varied backgrounds via augmentation and supplementary images) | Live drone or in-field camera capture |
| Disease classes | 10 classes: Early Blight, Late Blight, Leaf Mold, Septoria Leaf Spot, Bacterial Spot, Target Spot, Spider Mite damage, Tomato Mosaic Virus, Yellow Leaf Curl Virus, and Healthy | Pest identification beyond mite damage; soil, weather, or irrigation analysis |
| Deployment | Local inference application (Streamlit/Flask) runnable on a laptop or as a Progressive Web App | Native Android/iOS app store release; cloud-scale production deployment |
| Evaluation | Offline metrics + qualitative inference latency on CPU | Long-term field trial, multi-season validation |

The earlier wording — *"the dataset used … may be obtained … under controlled conditions"* — is **replaced** by the row above: the test split intentionally includes mixed-condition images so that the evaluation reflects the robustness goal stated in the Problem Statement. This removes the contradiction flagged in the alignment review.

---

## 4. Revised Section 3.3 — Dataset Description (Refined)

### 4.1 Sources

- **Primary source:** PlantVillage Tomato subset (~18,000 labeled images across the 10 target classes), obtained from Kaggle.
- **Supplementary source:** ~1,000–1,500 field-style images aggregated from open sources (Mendeley plant-pathology datasets, Roboflow Universe public projects). These contain natural backgrounds, multiple leaves per frame, and varied lighting.

### 4.2 Class List and Target Counts

| # | Class | Approx. images (post-augmentation) |
|---|---|---|
| 1 | Healthy | 2,000 |
| 2 | Early Blight | 2,000 |
| 3 | Late Blight | 2,000 |
| 4 | Leaf Mold | 2,000 |
| 5 | Septoria Leaf Spot | 2,000 |
| 6 | Bacterial Spot | 2,000 |
| 7 | Target Spot | 2,000 |
| 8 | Spider Mite damage | 2,000 |
| 9 | Tomato Mosaic Virus | 2,000 |
| 10 | Yellow Leaf Curl Virus | 2,000 |

### 4.3 Labeling

Images from PlantVillage and Roboflow arrive pre-labeled at the class level; no manual re-annotation is required. A spot-check (5 % stratified random sample) is performed manually to confirm label quality before training.

### 4.4 Split Strategy

- **Training:** 70 % — used to fit the CNN weights.
- **Validation:** 15 % — used for hyper-parameter tuning and early stopping.
- **Test (Controlled):** 10 % — held-out PlantVillage images, reports headline accuracy.
- **Test (Field-style):** 5 % — drawn exclusively from the supplementary mixed-condition images, reports generalization metric.

Reporting two test accuracies (controlled vs. field-style) directly addresses the generalization gap raised in the Problem Statement.

### 4.5 Class Imbalance Handling

Where a source class has fewer than 2,000 images, on-the-fly augmentation (rotation, horizontal/vertical flips, brightness/contrast jitter, random crop) is used during training rather than offline duplication. Class weights are also passed to the loss function so that under-represented classes are not dominated.

---

## 5. New Section 3.18 — Real-Time Inference Framework (NEW)

This section, missing from the original methodology, defines the **deployment pipeline** so the "real-time decision support" objective is concretely deliverable in Chapter 4.

### 5.1 Architecture Overview

```
[ Leaf image (upload / camera) ]
              |
              v
[ Preprocessing module: resize 224x224, normalize ]
              |
              v
[ Trained CNN (.h5 / .tflite) loaded once in memory ]
              |
              v
[ Softmax output -> top-1 class + confidence ]
              |
              v
[ Web UI: shows class label, confidence, and a short
  advisory note about the predicted disease ]
```

### 5.2 Target Platform

- **Primary:** Local web application built with **Streamlit** (Python 3.10+), runnable on any laptop with 4 GB RAM and no GPU.
- **Secondary (stretch):** Export the model to TensorFlow Lite (`.tflite`) so the same weights can be loaded by a mobile front-end in future work.

### 5.3 Performance Targets

| Metric | Target |
|---|---|
| Model size on disk | < 20 MB |
| Single-image CPU inference latency | < 1.5 s on a 4-core laptop |
| Controlled-test accuracy | ≥ 95 % |
| Field-style test accuracy | ≥ 85 % |

### 5.4 User Flow

1. User opens the local Streamlit page in a browser.
2. User uploads a tomato leaf photo (JPEG/PNG).
3. App returns predicted class, confidence percentage, and a short cached advisory paragraph for that disease.
4. Optionally, the user can capture a fresh image through the browser camera widget on supported devices.

### 5.5 Tools and Libraries

- **Modeling:** TensorFlow / Keras (transfer learning on MobileNetV2 or EfficientNet-B0).
- **Preprocessing:** OpenCV, NumPy, Albumentations.
- **Interface:** Streamlit.
- **Packaging:** Conda environment + `requirements.txt`; optional Docker image for reproducibility.
- **Hardware used for training:** Google Colab (T4 GPU); inference runs on CPU.

---

## 6. New Section 3.19 — Model Choice Justification (NEW)

The original methodology lists VGG16, ResNet50, and MobileNet as candidates but does not commit to one. To close the deployment gap:

- **Selected backbone:** **MobileNetV2** (primary) — chosen for its depthwise-separable convolutions, ~14 MB footprint, and proven track record on PlantVillage-style tasks.
- **Comparison backbone:** **EfficientNet-B0** — trained under the same protocol to provide a side-by-side accuracy/size comparison in Chapter 4.
- **Why not VGG16 / ResNet50?** Both exceed 90 MB and impose >2× the inference cost without delivering meaningfully higher accuracy on this task, which conflicts with the affordability requirement.

---

## 7. New Section 3.20 — Handling Real-World Variability (NEW)

To make the model generalize beyond the controlled PlantVillage backgrounds:

1. **Augmentation pipeline (training only):** random rotation (±30°), horizontal & vertical flips, brightness jitter (±20 %), contrast jitter (±20 %), random crop with resize, Gaussian blur (low probability).
2. **Mixed-condition validation:** the field-style supplementary images are excluded from training and used only for the field-style test split, giving an honest measure of generalization.
3. **Test-time check:** confusion matrices are computed separately on the controlled and field-style splits. A large gap between the two will be reported transparently in Chapter 4 as a known limitation rather than hidden by a single average.

---

## 8. Revised Chapter 4 Deliverables Checklist (Forward-Looking)

To confirm Chapter 4 readiness, the following artifacts will be produced:

- [ ] Cleaned and split dataset with documented class counts.
- [ ] Reproducible training notebook (Colab) for MobileNetV2 and EfficientNet-B0.
- [ ] Saved model files (`.h5` and `.tflite`).
- [ ] Evaluation report: accuracy, precision, recall, F1, confusion matrices for **both** controlled and field-style test splits.
- [ ] Streamlit inference application source code with a `README.md` explaining setup.
- [ ] Screenshots of the application performing inference on at least three classes.
- [ ] Discussion of failure cases (mis-classifications) with example images.

---

## 9. Summary of Changes vs. Original Write-Up

| Original gap | Where it appeared | How this revision resolves it |
|---|---|---|
| Scope said "controlled" but problem demanded real-world variability | §1.6 | §3 above redefines scope to include a field-style test split and an augmentation pipeline that simulates variability |
| "Real-time decision support" was named but not designed | Objective (v) and §3.16 | New §3.18 defines architecture, target platform, performance targets, and user flow |
| Accessibility / affordability not engineered | §1.2, §3.7 | New §3.19 commits to MobileNetV2 with explicit size and latency budgets |
| Dataset classes and split strategy vague | §3.3 | Revised §3.3 lists 10 classes, target counts, labeling, four-way split, and imbalance handling |

With these revisions, the Problem Statement, Aim, Objectives, Scope, and Methodology are internally consistent and the implementation phase has a concrete, testable specification.
