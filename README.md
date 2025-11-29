# 📊 Clustering Algorithms and Unsupervised Learning Techniques

This repository contains a comprehensive implementation of nine different unsupervised learning techniques, focusing primarily on clustering algorithms and related anomaly detection methods. Each section is implemented within a dedicated Google Colab notebook for reproducibility and clear documentation, fulfilling the requirements of the course assignment.

## 🚀 Project Overview

The core objective of this project was to explore and implement diverse clustering paradigms, ranging from fundamental centroid-based methods (K-Means) to density-based models (DBSCAN), probabilistic models (GMM), and applications in specialized domains like time-series analysis and multimodal embeddings (Text, Image, Audio).

| Notebook Status | Total |
| :--- | :---: |
| **Implemented & Documented** | 9 / 9 |
| **Algorithm Types** | Centroid-based, Density-based, Probabilistic, Time-series, Embedding-based |

---

## 🗂️ Notebook Directory

Below are the links to the nine implemented solutions, grouped by category.

| Section | Algorithm/Concept | Implementation Focus | Colab Link |
| :---: | :--- | :--- | :--- |
| **A** | **K-Means from Scratch** | Manual implementation using NumPy (E-step & M-step). | [K_Means_Implementation_From_Scratch.ipynb](https://colab.research.google.com/drive/1mAPjwZJj1qUfsuaA5ZHl6x2S9gsh994O) |
| **B** | **Optimal K-Means (External)** | Determining optimal cluster count using the **Elbow Method** (Inertia). | [K_Means_Elbow_Method_for_Optimal_K.ipynb](https://colab.research.google.com/drive/1D3f-da1PfXnIIW7LpKvSNJoUuWDLojX5) |
| **C** | **Gaussian Mixture Models (GMM)** | Probabilistic clustering, soft assignments, and covariance. | [GMM_Clustering_Probabilistic_Approach.ipynb](https://colab.research.google.com/drive/1jDIIjypp5cJa3mH6dC6v_fyOnEaIa9ta) |
| **D** | **DB Scan Clustering** | Density-based clustering (Core/Border/Noise points) using **PyCaret**. | [DBSCAN_Density_Based_Clustering.ipynb](https://colab.research.google.com/drive/1FMjwuDZ-DCuSbvPov7kWZqYM_ql9ftjd#scrollTo=IpAmAKG5wEaQ) |
| **E** | **Anomaly Detection (PyOD)** | Outlier detection using **Isolation Forest** (IForest) from the PyOD library. | [Anomaly_Detection_with_PyOD_IForest.ipynb](https://colab.research.google.com/drive/1uLqymrY9r2_S9utcPmCi4cTpe9AJnECs#scrollTo=DlgAFyTcxjeh) |
| **F** | **Time-series Clustering** | Clustering time-series data using **K-Shape** and the **DTW** concept. | [Timeseries_Clustering_with_DTW.ipynb](https://colab.research.google.com/drive/1fwuF4r6pY21J-e-990mtz8Sf7ilIBKeK#scrollTo=kdyX3HjAzFER) |
| **G** | **Document Clustering** | Clustering text documents based on **LLM Embeddings** (Sentence Transformers). | [Document_Clustering_LLM_Embeddings.ipynb](https://colab.research.google.com/drive/1Y2EW5DnwjyDZ-A6crny62khMog6AEZQm#scrollTo=lk3ZPfV2zoKX) |
| **H** | **Image Clustering** | Conceptual implementation of clustering **Image Embeddings** (e.g., CLIP/ImageBind). | [Image_Clustering_Using_Embeddings.ipynb](https://colab.research.google.com/drive/1NGvL6hAHC33r_K1mP8P7U65n_2OxXPyr#scrollTo=dw-mqSMy2DwY) |
| **I** | **Audio Clustering** | Conceptual implementation of clustering **Audio Embeddings** (e.g., VGGish/ImageBind). | [Audio_Clustering_Using_Embeddings.ipynb](https://colab.research.google.com/drive/1vln6-6v6v1hgWWk3YZSKdIe5fbApOuUl#scrollTo=_ijO7juE2iZ1) |

---

## 🔑 Key Concepts Covered

* **Metric Choice:** Euclidean Distance vs. Dynamic Time Warping (DTW).
* **Initialization:** Random initialization and its impact on K-Means.
* **Model Selection:** Elbow Method for $K$ and contamination factor for outliers.
* **Feature Engineering:** Converting raw data (Text, Time-series) into clusterable **vector embeddings**.

## 🛠️ Required Environment

* Python 3.x
* The notebooks utilize the following primary libraries: `numpy`, `scikit-learn`, `matplotlib`, `pycaret`, `pyod`, `tslearn`, and `sentence-transformers`.
* All necessary installations are provided within the corresponding Google Colab notebooks.

---
