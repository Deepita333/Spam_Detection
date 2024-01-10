# Spam_Detection
A ml model based on binary classification that classifies your email as spam and ham (not spam).
A ml model based on binary classification to classify the emails into two categories as spam and not spam(ham)✉️
Dataset used is -https://www.kaggle.com/code/ardava/spam-email-classification-using-random-forest🔗 Have added comments in the project to help you understand it one step at a time😁
```
import nltk #we will be using natural language toolkit
nltk.download('stopwords') #used to remove the stopwords like 'a' , 'an', 'the','and'

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import string
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier




# Read the CSV file
df = pd.read_csv("D:\spam_ham_dataset.csv")
df.replace({r'\r\n':' '}, regex=True, inplace=True)

# Handle NaN values in 'text' column
df = df.dropna(subset=['label_num']) #dropping the rows wd nan values as I faced errors

# Text preprocessing
ps = PorterStemmer() #tool to reduce words to their root-form . Example-'playing' becomes 'play'
res = []

stop_words = set(stopwords.words('english'))
stop_words.remove('not')

for i in range(len(df)):
    text = str(df['text'][i]).lower().translate(str.maketrans('', '', string.punctuation)).split()
    text = [ps.stem(word) for word in text if word not in stop_words]
    text = ' '.join(text)
    res.append(text)

# Vectorize the text
cv = CountVectorizer(max_features=42500) #CountVectorizer converts text into numerical representations for machine learning
X = cv.fit_transform(res).toarray()
y = df['label_num']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Define the model_score function before using it
def model_score(y_true, y_pred):
    acc_scor = accuracy_score(y_true, y_pred)
    prec_scor = precision_score(y_true, y_pred)
    recall_scor = recall_score(y_true, y_pred)
    f1_scor = f1_score(y_true, y_pred)
    overall_avg_score = (acc_scor + prec_scor + recall_scor + f1_scor) / 4

    print(f'Model accuracy score: {acc_scor}')
    print(f'Model precision score: {prec_scor}')
    print(f'Model recall score: {recall_scor}')
    print(f'Model f1 score: {f1_scor}')
    print(f'Average overall score performance: {overall_avg_score}')

    print(confusion_matrix(y_true, y_pred)) #table to summarize the no. of correct and incorrect predictions in each row

# Choose and train a model (Naive Bayes) #Naive Bayes is a powerful tool in ML used for spam filtering , text classification , sentiment analysis
model = MultinomialNB()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate performance
model_score(y_test, y_pred)
#Random Forest Classifier
# Create and train a Random Forest classifier
cl_rf = RandomForestClassifier(n_estimators=100, random_state=42)
cl_rf.fit(X_train, y_train)

# Make predictions using Random Forest
y_pred = cl_rf.predict(X_test)

# Evaluate performance of Random Forest 
model_score(y_test, y_pred)
np.column_stack((y_test[:15], y_pred[:15]))

#text preprocessing is followed as it was followed in case of training data
def preprocess_email(email_text):
     # Convert to lowercase
    email_text = email_text.lower()

    # Remove punctuation
    email_text = email_text.translate(str.maketrans('', '', string.punctuation))

    # Remove stop words
    words = email_text.split()
    words = [word for word in words if word not in stop_words]

    # Apply stemming 
    words = [ps.stem(word) for word in words]  

    # Join back the words
    processed_text = ' '.join(words)

    return processed_text
email_text = input("Enter the text of the email you want to check: ")
new_email_text = preprocess_email(email_text)  # Preprocess the text
new_email_vec = cv.transform([new_email_text]).toarray()

# Choose model and make prediction 
prediction = cl_rf.predict(new_email_vec)[0]

if prediction == 0:
    print("The email is likely not spam (ham).")
else:
    print("The email is likely spam.")

```


