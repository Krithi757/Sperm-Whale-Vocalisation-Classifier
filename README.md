# Sperm Whale Vocalization (Coda) Classifier

An end-to-end data engineering and deep learning preprocessing pipeline designed to analyze, clean, and format acoustic sequence data from sperm whale vocalizations. This project focuses on the structural and rhythmic classification of clicks—often compared to human linguistic patterns or "vowels"—emitted during social interactions.

This repository serves as a core framework for bioacoustic language modeling and next-token sequential prediction.

## 📌 Project Overview

Sperm whales communicate using short, rhythmic patterns of clicks known as **codas**. To study their underlying grammar and communication structure, this project implements a rigorous data preprocessing architecture. The system converts raw acoustic timelines and multi-channel features into clean, high-performance tensors ready for sequential deep learning architectures like Long Short-Term Memory (LSTM) networks or Transformers.

---

## 🛠️ Data Engineering & Preprocessing Pipeline

The pipeline follows a strict production-grade workflow to ensure data integrity and prevent downstream errors:

### 1. Quality Control & Completeness Audit
* **Null Value Scanning:** The dataset maintains $100\%$ data completeness with zero missing or null entries across all $8,719$ chronological records.
* **Sequential Duplicate Verification:** Verified $0$ identical row anomalies, confirming that each entry represents a unique, independent acoustic event occurring in continuous time.

### 2. Statistical Range & Outlier Analysis
Instead of blindly removing mathematical anomalies, an Inter-Quartile Range (IQR) analysis was conducted to study the acoustic diversity:
* **Formula Applied:** $$IQR = Q_3 - Q_1$$
  $$\text{Lower Bound} = Q_1 - 1.5 \times IQR$$
  $$\text{Upper Bound} = Q_3 + 1.5 \times IQR$$
* **Domain Adaptation:** Statistical outliers flagged in click counts (`nClicks`) and Inter-Click Intervals (`ICI5` to `ICI9`) were deliberately **retained**. Because a small percentage of calls naturally span longer click lengths ($6$ to $10$ clicks), dropping these entries would artificially decimate the vocal vocabulary. 

### 3. Vocabulary Standardization & Label Encoding
* The raw acoustic vocabulary consists of $35$ unique, multi-click string tokens (e.g., `'1+1+3'`, `'5R1'`, `'5R3'`).
* These strings are mapped alphabetically to a strict range of integer identifiers from $0$ to $34$ (`CodaTokenID`). This allows a deep learning network's Embedding Layer to compute spatial relationships seamlessly.
* Biological metadata (such as whale `Clan`, social `Unit`, and individual identifier `IDN`) are preserved as contextual references for downstream model evaluation.

### 4. Sliding Window Sequential Framing
To model the conversational stream as a next-token prediction task, the timeline is segmented into a moving window framework:
* **Context Length ($X$):** $5$ historical tokens are analyzed sequentially.
* **Target Objective ($y$):** The $6\text{th}$ consecutive token is extracted as the target label.
* **Transformation Grid:** Converts a one-dimensional array into an input history matrix of shape `(8714, 5)`.

### 5. Leakage-Proof Chronological Dataset Splitting
Traditional random shuffling causes severe **temporal data leakage** because overlapping conversational histories blend across training and evaluation boundaries. To ensure rigorous validation, the data is sliced strictly in chronological order:
* **🔮 Training Set ($80\%$):** $6,971$ sequence frames for pattern baseline learning.
* **🧪 Validation Set ($10\%$):** $871$ sequence frames used for active hyperparameter tuning.
* **🏁 Testing Set ($10\%$):** $872$ sequence frames kept in isolation for the final model examination.

---

## 📊 Dataset Architecture

The master matrix (`DominicaCodas_Cleaned.csv`) preserves the following features:

| Column Name | Data Type | Description |
| :--- | :--- | :--- |
| `codaNUM2018` | `int64` | Unique chronological sequence index identifier |
| `nClicks` | `int64` | Total number of individual pulses within the coda |
| `Duration` | `float64` | Complete physical time span of the vocalization (seconds) |
| `ICI1` to `ICI9` | `float64` | Inter-Click Intervals measuring silence gaps between pulses |
| `CodaType` | `object` | Original string representation of the acoustic pattern |
| `CodaTokenID` | `int64` | Clean target integer mapping ($0$ to $34$) for model training |
| `Clan`, `Unit`, `IDN`| `object` | Biological metadata tracking whale language groups and family units |

---

## 🚀 Getting Started

### Prerequisites
Ensure your local environment or notebook environment has the following core data science toolkits installed:
```bash
pip install numpy pandas
