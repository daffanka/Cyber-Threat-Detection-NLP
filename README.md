# KBJ - Cyber Threat Detection: BERT vs Traditional ML

![Python](https://img.shields.io/badge/Python-3.x-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-Enabled-red)
![Status](https://img.shields.io/badge/Status-Completed-green)

## 📌 Overview
This project focuses on **Cyber Threat Intelligence (CTI)** classification using Natural Language Processing (NLP). We aim to detect threat-related descriptions in text data by comparing the performance of a **Fine-Tuned BERT** model against traditional Machine Learning algorithms (**Random Forest, SVM, XGBoost**) using BERT-generated embeddings.

The objective is to accurately classify text into **Attack** (Malware, Exploits, Threats) vs **Non-Attack**.

## 🚀 Methodology
The pipeline consists of two main approaches:

1.  **Deep Learning Approach**:
    * **Model**: Fine-Tuning `bert-base-uncased`.
    * **Architecture**: BERT Encoder + Linear Classification Head.
    * **Training**: Batched training with PyTorch to optimize VRAM usage.

2.  **Hybrid Approach (Embeddings + ML)**:
    * **Embedding**: Generating high-quality text embeddings using `sentence-transformers/all-MiniLM-L6-v2`.
    * **Classifiers**: Random Forest, SVM (RBF Kernel), and XGBoost.

## 📂 Project Structure

    ├── data/            # Dataset (Train/Val/Test splits)
    ├── notebooks/       # Jupyter Notebook containing the full pipeline
    ├── requirements.txt # Dependencies list
    └── README.md        # Project documentation

## 📊 Results Snapshot

Comparison of metrics on the Validation Set.

The results demonstrate that **BERT (Fine-Tuned)** generally outperforms traditional Machine Learning models, particularly in **Accuracy** and **F1-Score**, proving its superior capability in capturing complex semantic patterns in cyber threat descriptions.

| Model                 | Accuracy | Precision | Recall   | F1-Score |
|-----------------------|----------|-----------|----------|----------|
| **Random Forest** | 0.7705   | 0.4788    | 0.7548   | 0.5859   |
| **SVM** | 0.7945   | 0.5141    | **0.8121** | 0.6296   |
| **XGBoost** | 0.7781   | 0.4904    | **0.8121** | 0.6115   |
| **BERT (Fine-Tuned)** | **0.8240** | **0.5632** | 0.8089   | **0.6641** |

### Key Observation
* **Best Overall Performance**: **BERT** achieved the highest **F1-Score (0.6641)** and **Accuracy (0.8240)**, indicating a better balance between precision and recall compared to traditional models.
* **Recall vs Precision**: While SVM and XGBoost achieved marginally higher Recall (~0.81), they suffered significantly in Precision (~0.49-0.51). BERT maintained a comparable Recall (~0.81) but with much better Precision (~0.56), leading to fewer False Positives.

## 🛠️ Installation & Usage

1.  **Clone the repository**
    ```bash
    git clone [https://github.com/USERNAME_ANDA/Cyber-Threat-Detection.git](https://github.com/daffanka/Cyber-Threat-Detection-NLP)
    cd Cyber-Threat-Detection
    ```

2.  **Install dependencies**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the Notebook**
    Open `notebooks/Threat_Detection.ipynb` in Jupyter Notebook or Google Colab.

## 🤝 Acknowledgments
* **Dataset**: Private Cyber Threat Intelligence Dataset (Splited Train/Val/Test).
* **Libraries**: Hugging Face Transformers, Scikit-Learn, PyTorch.