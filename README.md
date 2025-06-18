
<h2 align="center"><span style="font-family: Babas; font-size: 2em;">E-commerce Text Classification</span></h2>
<h4 align="center"><span style="font-family: Babas; font-size: 1.5em; font-style: italic">Extended with Sentence-BERT, LoRA, and LSTM Experiments</span></h4>
<h4 align="center"><span style="font-family: Babas; font-size: 1.5em;">Originally by Sugata Ghosh · Extended by Gökalp Eren Akol</span></h4>

---

### Project Overview

The objective of this project is to classify [**e-commerce**](https://en.wikipedia.org/wiki/E-commerce) products into four categories based on their textual descriptions:

- `Electronics`
- `Household`
- `Books`
- `Clothing & Accessories`

This repository is based on the original work by [**Sugata Ghosh**](https://github.com/sugatagh), who built the foundational pipeline using **TF-IDF** and **Word2Vec** for feature extraction and machine learning classifiers for prediction.

This extended version builds upon that work by integrating **Sentence-BERT embeddings**, **LoRA-based fine-tuning**, and **LSTM neural networks** to explore more advanced NLP techniques and improve classification accuracy.

---

Data Sources:

Word2Vec: https://www.kaggle.com/code/sugataghosh/e-commerce-text-classification-tf-idf-word2vec?select=GoogleNews-vectors-negative300.bin

E-commerce Dataset: https://www.kaggle.com/datasets/saurabhshahane/ecommerce-text-classification/data

###  Baseline Contributions by Sugata Ghosh

- Exploratory data analysis on character/word counts

  ![image](https://github.com/user-attachments/assets/165b4d9a-74de-4a48-a899-6054947352c4)

- Text normalization and TF-IDF vectorization
- Classical ML classification (SVM, etc.)
- Word2Vec embeddings with XGBoost

  ![image](https://github.com/user-attachments/assets/3e28f747-f874-4362-81d1-848dda6d63c6)

- Achieved ~94.8% test accuracy

Baseline GitHub Repository: [https://github.com/sugatagh](https://github.com/sugatagh)

---

### Additional Contributions 
![image](https://github.com/user-attachments/assets/5677b3f1-29ba-4b05-afbf-7575322a7b45)

#### 1. **Sentence-BERT Embeddings + ML**

- Used [`all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) model to generate semantically meaningful sentence embeddings.
- Embedded product descriptions were fed into classic classifiers.
- Best result: **K-Nearest Neighbors** (KNN) with hyperparameter tuning → **96% validation accuracy**

#### 2. **LoRA Fine-Tuning on SBERT**

- Applied **Low-Rank Adaptation (LoRA)** to fine-tune only a small portion of SBERT model parameters.
- After 3 training epochs, achieved **92.6%** validation accuracy.
- Switched to a larger model (`all-mpnet-base`) for better performance → **94.4% accuracy**

  ![image](https://github.com/user-attachments/assets/6085a545-067e-4ff1-9afd-c4a355747afa)


  ![image](https://github.com/user-attachments/assets/8dcee279-8483-4523-8a57-ca4c5096ef08)

#### 3. **LSTM Neural Network Experiments**

- Trained a basic **LSTM** model using PyTorch’s `nn.Embedding` layer → **87% accuracy**
- Enhanced LSTM by replacing its input with **SBERT embeddings**
- Improved performance to **~95%**, making it competitive with traditional ML methods



SBERT + LSTM Validation Accuracies

![image](https://github.com/user-attachments/assets/fa101a4d-c59e-4c5f-a8ff-319b0539096b)


---

### Accuracy Summary

![image](https://github.com/user-attachments/assets/67041c3b-01ae-4798-8b19-13a61e946948)


| Approach                           | Model                         | Validation Accuracy |
|------------------------------------|-------------------------------|---------------------|
| TF-IDF + SVM                       | Linear SVM                    | 95.0%               |
| Word2Vec + SVM                     | Context embeddings            | 94.9%               |
| SBERT Embeddings + KNN             | all-MiniLM-L6-v2              | **96.0%**           |
| SBERT LoRA Fine-Tuning             | all-MiniLM-L6-v2              | 92.6%               |
| SBERT Fine-Tune (larger model)     | all-mpnet-base                | 94.4%               |
| LSTM + Torch Embedding             | Basic RNN                     | 87.0%               |
| LSTM + SBERT Embeddings            | Hybrid Architecture           | 95.0%               |


---
