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
 Evaluation Evaluate the model performances and compared with approach 1

