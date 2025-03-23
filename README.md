# Text-Processing-and-Prediction ,Sentiment Analysis of IMDB Dataset

## 1. Introduction
This project is based on sentiment analysis of the IMDB dataset using different text processing approaches, training a model for sentiment classification, and evaluating its performance.  

Three approaches were used:
- **Classification Model using Text Features**  
- **Model Applying Stop Word Processing**  
- **Fine-Tuned Embedding Model**  

These models were evaluated using classification accuracy, precision, recall, and F1-score.

---

## 2. Steps of the Three Approaches

### 2.1 Approach 1: Classification Model using Text Features
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

### 2.2 Approach 2: Model Applying Stop Word Processing
- Applied the same text preprocessing techniques as in Approach 1.  
- Additionally, removed stop words using:  
  ```python
  TfidfVectorizer(stop_words='english')
- Repeating the feature extraction process
- Model Training- Using newly transformed TF-IDF dataset, another Logistic Regression model was trained.
- Evaluate the model performances and compared with approach 1

---

### 2.3 Approach 3: Fine-TunedEmbeddingModel-(FastText Supervised Learning Model)
- Applied the same text preprocessing techniques as in Approach 2.
- Prepare Data for FastText- Generated a file where each line included a label, which starts with  __label__ followed by the processed text.
- Model Training - Using parameters (epoch=25, lr=0.5, wordNgrams=2), Fast Text model was trained
- Prediction and Evaluation- Trained model was used to predict sentiment, on test and validation dataset and
 analyzed the results

---

## 3. Analysis of the Results
### 3.1 Results of Model 1 - TF-IDF (With Stop Words)
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
