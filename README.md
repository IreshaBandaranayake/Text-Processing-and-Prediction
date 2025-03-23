# Text-Processing-and-Prediction 
(Sentiment Analysis of IMDB Dataset)

This project is based on sentiment analysis of the IMDB dataset using different text processing approaches, training a model for sentiment classification, and evaluating its performance.  

Three approaches were used:
- **Classification Model using Text Features**  
- **Model Applying Stop Word Processing**  
- **Fine-Tuned Embedding Model**  

These models were evaluated using classification accuracy, precision, recall, and F1-score.

---

## Steps of the Three Approaches

### Approach 1: Classification Model using Text Features
**Text Preprocessing**  
The IMDB dataset contains movie reviews labeled as either positive or negative. Before applying to the model, the dataset was preprocessed:  
- **Removing Punctuations and Extra Spaces**: Replaced all punctuations with spaces except a-z and A-Z, then removed extra spaces.  
- **Text Lowercasing**: Converted all text to lowercase.  
- **Tokenization**: Split text into single words using `nltk.word_tokenize()`.  

**Feature Extraction**  
Raw text was converted into numerical TF-IDF features using `TfidfVectorizer`, considering the top 10,000 most important words.  

**Model Training**  
A Logistic Regression model was trained using the TF-IDF transformed dataset.  

**Evaluation**  
The trained model was evaluated using a test dataset.

---

### Approach 2: Model Applying Stop Word Processing
- Applied the same text preprocessing techniques as in Approach 1.  
- Additionally, removed stop words using:  
  ```python
  TfidfVectorizer(stop_words='english')
- Repeating the feature extraction process
- Model Training- Using newly transformed TF-IDF dataset, another Logistic Regression model was trained.
- Evaluate the model performances and compared with approach 1

---

### Approach 3: Fine-TunedEmbeddingModel-(FastText Supervised Learning Model)
- Applied the same text preprocessing techniques as in Approach 2.
- Prepare Data for FastText- Generated a file where each line included a label, which starts with 
  `__label__` followed by the processed text.
- Model Training - Using parameters (epoch=25, lr=0.5, wordNgrams=2), Fast Text model was trained
- Prediction and Evaluation- Trained model was used to predict sentiment, on test and validation dataset and
 analyzed the results

---

## Analysis of the Results
### Results of Model 1 - TF-IDF (With Stop Words)
#### Test Dataset
- Accuracy: 0.8928
- Precision: 0.90 (Negative), 0.88 (Positive)
- Recall: 0.88 (Negative), 0.90 (Positive)
- F1-score: 0.89 (Overall)

#### Validation Dataset
- Accuracy: 0.89
- Precision: 0.90 (Negative), 0.88 (Positive)
- Recall: 0.87 (Negative), 0.91 (Positive)
- F1-score: 0.89 (Overall)

---
### Results of Mode2 - TF-IDF (With Stop Words)
#### Test Dataset
- Accuracy: 0.894
- Precision: 0.90 (Negative), 0.88 (Positive)
- Recall: 0.88 (Negative), 0.91 (Positive)
- F1-score: 0.89 (Overall)

#### Validation Dataset
- Accuracy: 0.8866
- Precision: 0.90 (Negative), 0.87 (Positive)
- Recall: 0.87 (Negative), 0.90 (Positive)
- F1-score: 0.89 (Overall)

---
### Results of Mode3 - Fast Text Embedding Model
#### Test Dataset
- Accuracy: 0.9072
- Precision: 0.91 (Negative), 0.90 (Positive)
- Recall: 0.90 (Negative), 0.92 (Positive)
- F1-score: 0.91 (Overall)
  
#### Validation Dataset
- Accuracy: 0.9058
- Precision: 0.91 (Negative), 0.90 (Positive)
- Recall: 0.90 (Negative), 0.91 (Positive)
- F1-score: 0.90 (Overall)

---
###
Three different approaches were applied for sentiment analysis in this study.

- TF-IDF models provided acceptable accuracy (~89%), with stop-word removal showing only slight improvement.

- FastText performed the best, achieving the highest accuracy (90.7%) and demonstrating superior ability in identifying word meanings, handling spelling mistakes, and recognizing unseen words.

- FastText is the preferred choice for this task due to its word embedding capabilities and robustness in sentiment analysis.
  
---
### How to Run
1. Clone the repository:
  `git clone https://github.com/IreshaBandaranayake/Text-Processing-and-Prediction`


2. Install required dependencies:
   `pip install nltk fasttext sklearn`

3. Run the sentiment analysis script:
   `python sentiment_analysis.py`
