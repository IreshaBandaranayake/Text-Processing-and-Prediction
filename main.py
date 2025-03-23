import pandas as pd
import re
import nltk
import fasttext
import fasttext.util
from  nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer #convert raw text into numerical TF-IDF features
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
from stop_words import get_stop_words

nltk.download('punkt')

# Import Dataset
train_dataset = pd.read_csv('Train.csv')
test_dataset = pd.read_csv('Test.csv')
valid_dataset = pd.read_csv('Valid.csv')

#Loading Stop Words
stop_words = set(get_stop_words('english'))

# Text preprocessing
def clean_text(raw_text):
    raw_text = re.sub('[^a-zA-Z]', ' ', raw_text)  # replace all other punctuations with space except a-z and A-Z
    raw_text = re.sub('\\s+', ' ', raw_text).strip()  # remove extra spaces
    raw_text = raw_text.lower()  # convert capitals to lowercase
    tokens = word_tokenize(raw_text) #tokenization
    return " ".join(tokens) #join tokens into a String

#Applying preprocessing to datasets
train_dataset["processed_text"] = train_dataset["text"].apply(clean_text)
test_dataset["processed_text"] = test_dataset["text"].apply(clean_text)
valid_dataset["processed_text"] = valid_dataset["text"].apply(clean_text)

#Approach One - TF-IDF without removing stopwords

tfidf_obj1 = TfidfVectorizer(max_features=10000)
train_text_1 = tfidf_obj1.fit_transform(train_dataset['processed_text'])
test_text_1 = tfidf_obj1.transform(test_dataset['processed_text'])
valid_text_1 = tfidf_obj1.transform(valid_dataset['processed_text'])

#Training Logistic Regression Model
LR_Model1 = LogisticRegression()
LR_Model1.fit(train_text_1, train_dataset['label'])
LR_predict_1_test = LR_Model1.predict(test_text_1)
LR_predict_1_valid = LR_Model1.predict(valid_text_1)
print('Model 1 - Performances for Test Dataset')
print('Accuracy Score:', accuracy_score(test_dataset['label'], LR_predict_1_test))
print("Classification Report:\n", classification_report(test_dataset['label'], LR_predict_1_test))

print('Model 1 - Performances for Validation Dataset')
print('Accuracy Score:', accuracy_score(valid_dataset['label'], LR_predict_1_valid))
print("Classification Report:\n", classification_report(valid_dataset['label'], LR_predict_1_valid))

#Approach Two - TF-IDF with stopwords removing

tfidf_obj2 = TfidfVectorizer(max_features=10000, stop_words='english')
train_text_2 = tfidf_obj2.fit_transform(train_dataset['processed_text'])
test_text_2 = tfidf_obj2.transform(test_dataset['processed_text'])
valid_text_2 = tfidf_obj2.transform(valid_dataset['processed_text'])

#Training Logistic Regression Model
LR_Model2 = LogisticRegression()
LR_Model2.fit(train_text_2, train_dataset['label'])
LR_predict_2_test = LR_Model2.predict(test_text_2)
LR_predict_2_valid = LR_Model2.predict(valid_text_2)

print('Model 2 - Performances for Test Dataset')
print('Accuracy Score:', accuracy_score(test_dataset['label'], LR_predict_2_test))
print("Classification Report:\n", classification_report(test_dataset['label'], LR_predict_2_test))

print('Model 2 - Performances for Validation Dataset')
print('Accuracy Score:', accuracy_score(valid_dataset['label'], LR_predict_2_valid))
print("Classification Report:\n", classification_report(valid_dataset['label'], LR_predict_2_valid))


#Approach Three - FAST TEXT Model
def create_fasttext_file(df, ft_file):
    with open(ft_file, 'w', encoding='utf-8') as f:
        for _, row in df.iterrows():
            f.write(f"__label__{row['label']} {row['processed_text']}\n")

#Create train and validate files
create_fasttext_file(train_dataset, 'ft_train_data.txt')
create_fasttext_file(valid_dataset, 'ft_valid_data.txt')

#FT model Training
ft_classifier = fasttext.train_supervised(
    input='ft_train_data.txt', epoch=25, lr=0.5, wordNgrams=2, verbose=2
)

#FT Model Evaluation
def predict_wth_ft(classifier, input_text):
    return [int(classifier.predict(text)[0][0].replace('__label__', '')) for text in input_text]

#Evaluate with Test Data
ft_prediction_wt_test = predict_wth_ft(ft_classifier, test_dataset['processed_text'].tolist())

print('Model 3 - Fast Text Performances for Test Dataset')
print('Accuracy', accuracy_score(test_dataset['label'], ft_prediction_wt_test))
print("Classification Report:\n", classification_report(test_dataset['label'], ft_prediction_wt_test))

#Evaluate with Validation Data
ft_prediction_wt_valid = predict_wth_ft(ft_classifier, valid_dataset['processed_text'].tolist())

print('Model 3 - Fast Text Performances for Valid Dataset')
print('Accuracy', accuracy_score(valid_dataset['label'], ft_prediction_wt_valid))
print("Classification Report:\n", classification_report(valid_dataset['label'], ft_prediction_wt_valid))
