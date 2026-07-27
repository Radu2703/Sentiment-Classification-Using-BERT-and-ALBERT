#Sentiment Classification using BERT and ALBERT

import os
import time
import tensorflow as tf
from transformers import BertTokenizer, TFBertForSequenceClassification, AlbertTokenizer, TFAlbertForSequenceClassification
import pandas as pd
from bs4 import BeautifulSoup
import re
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
from sklearn.metrics import classification_report

# Get the current working directory
current_folder = os.getcwd()
timer = time.time()
dataset = tf.keras.utils.get_file(fname ="aclImdb.tar.gz", origin ="http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz", cache_dir = current_folder, extract = True)
timer = time.time() - timer
print("\nTotal time downloading the dataset:", timer, "seconds\n")
dataset_path = os.path.dirname(dataset)
os.listdir(dataset_path)
dataset_dir = os.path.join(dataset_path, 'aclImdb')
os.listdir(dataset_dir)
train_dir = os.path.join(dataset_dir, 'train')
os.listdir(train_dir)

def load_dataset(directory):
	data = {"sentence": [], "sentiment": []}
	for file_name in os.listdir(directory):
		if file_name == 'pos':
			positive_dir = os.path.join(directory, file_name)
			for text_file in os.listdir(positive_dir):
				text = os.path.join(positive_dir, text_file)
				with open(text, "r", encoding="utf-8") as f:
					data["sentence"].append(f.read())
					data["sentiment"].append(1)
		elif file_name == 'neg':
			negative_dir = os.path.join(directory, file_name)
			for text_file in os.listdir(negative_dir):
				text = os.path.join(negative_dir, text_file)
				with open(text, "r", encoding="utf-8") as f:
					data["sentence"].append(f.read())
					data["sentiment"].append(0)
	return pd.DataFrame.from_dict(data)

# Load the training dataset
timer = time.time() - timer
train_df = load_dataset(train_dir)
print("\nThe first elements of the training dataset are:\n", train_df.head())
print()
# Load the testing dataset
test_dir = os.path.join(dataset_dir,'test')
test_df = load_dataset(test_dir)
print("\nThe first elements of the testing dataset are:\n", test_df.head())
timer = time.time() - timer
print("\nTotal time loading the datasets:", timer, "seconds\n")

# Show the bar chart
sentiment_counts = train_df['sentiment'].value_counts()
plt.bar(['Negative', 'Positive'], sentiment_counts.values)
plt.title('Sentiments Counters')
plt.xlabel('Sentiments')
plt.ylabel('Counters')
plt.show()

def text_cleaning(text):
	soup = BeautifulSoup(text, "html.parser")
	text = re.sub(r'\[[^]]*\]', '', soup.get_text())
	pattern = r"[^a-zA-Z0-9\s,']"
	text = re.sub(pattern, '', text)
	return text

timer = time.time()
# Training dataset
train_df['Cleaned_sentence'] = train_df['sentence'].apply(text_cleaning).tolist()
# Testing dataset
test_df['Cleaned_sentence'] = test_df['sentence'].apply(text_cleaning).tolist()
timer = time.time() - timer
print("\nTotal time cleaning the datasets:", timer, "seconds\n")

# Function to generate word cloud
def generate_wordcloud(text, Title):
	all_text = " ".join(text)
	wordcloud = WordCloud(stopwords=set(STOPWORDS)).generate(all_text)
	plt.figure(figsize=(10, 5))
	plt.imshow(wordcloud, interpolation='bilinear')
	plt.axis("off")
	plt.title(Title)
	plt.show()

timer = time.time()
positive = train_df[train_df['sentiment']==1]['Cleaned_sentence'].tolist()
generate_wordcloud(positive,'Positive Reviews WordCloud')
negative = train_df[train_df['sentiment']==0]['Cleaned_sentence'].tolist()
generate_wordcloud(negative,'Negative Reviews WordCloud')
timer = time.time() - timer
print("\nTotal time creating the wordclouds:", timer, "seconds\n\n")

# Training data
Reviews = train_df['Cleaned_sentence']
Target = train_df['sentiment']
# Testing data
test_reviews = test_df['Cleaned_sentence']
test_targets = test_df['sentiment']
x_test = test_reviews
y_test = test_targets



print("\nALBERT Model\n")
#Tokenize and encode the data using the ALBERT tokenizer
tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2', do_lower_case=True)
max_len = 128
# Tokenize and encode the sentences
timer = time.time()
X_train_encoded = tokenizer.batch_encode_plus(Reviews.tolist(), padding=True, truncation=True, max_length = max_len, return_tensors='tf')
X_test_encoded = tokenizer.batch_encode_plus(x_test.tolist(), padding=True, truncation=True, max_length = max_len, return_tensors='tf')
timer = time.time() - timer
print("\nTotal time tokenizing and encoding the sentences with the ALBERT model:", timer, "seconds\n")
print('\nALBERT model training review example:', Reviews[0])
print('\nALBERT model training sentiment example:', Target[0])

# Initialize the model
timer = time.time()
model = TFAlbertForSequenceClassification.from_pretrained('albert-base-v2', num_labels=2)
# Compile the model with an appropriate optimizer, loss function, and metrics
optimizer = tf.keras.optimizers.Adam(learning_rate=2e-5)
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')
model.compile(optimizer=optimizer, loss=loss, metrics=[metric])
timer = time.time() - timer
print("\nTotal time initializing and compiling the ALBERT model:", timer, "seconds\n")

# Train the model
timer = time.time()
history = model.fit([X_train_encoded['input_ids'], X_train_encoded['token_type_ids'], X_train_encoded['attention_mask']], Target, batch_size=32, epochs=3)
timer = time.time() - timer
print("\nTotal time training the ALBERT model:", timer, "seconds\n")

# Evaluate the model on the test data
timer = time.time()
test_loss, test_accuracy = model.evaluate([X_test_encoded['input_ids'], X_test_encoded['token_type_ids'], X_test_encoded['attention_mask']], y_test)
print('\nALBERT model test accuracy:', test_accuracy)
print('\nALBERT model test loss:', test_loss)
timer = time.time() - timer
print("\nTotal time evaluating the ALBERT model:", timer, "seconds\n")
timer = time.time()
pred = model.predict([X_test_encoded['input_ids'], X_test_encoded['token_type_ids'], X_test_encoded['attention_mask']])
# pred is of type TFSequenceClassifierOutput
logits = pred.logits
# Use argmax along the appropriate axis to get the predicted labels
pred_labels = tf.argmax(logits, axis=1)
# Convert the predicted labels to a NumPy array
pred_labels = pred_labels.numpy()
label = {1: 'Positive',	0: 'Negative'}
timer = time.time() - timer
print("\nTotal time predicting with the ALBERT model:", timer, "seconds")

# Map the predicted labels to their corresponding strings using the label dictionary
pred_labels = [label[i] for i in pred_labels]
Actual = [label[i] for i in y_test]
print('\nALBERT model predicted sentiments:', pred_labels[:10])
print('\nALBERT model actual sentiments:', Actual[:10])
print('\nALBERT model classification report:\n\n', classification_report(Actual, pred_labels),'\n\n')



print("\nBERT Model\n")
#Tokenize and encode the data using the BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
max_len = 128
# Tokenize and encode the sentences
timer = time.time()
X_train_encoded = tokenizer.batch_encode_plus(Reviews.tolist(), padding=True, truncation=True, max_length = max_len, return_tensors='tf')
X_test_encoded = tokenizer.batch_encode_plus(x_test.tolist(), padding=True, truncation=True, max_length = max_len, return_tensors='tf')
timer = time.time() - timer
print("\nTotal time tokenizing and encoding the sentences with the BERT model:", timer, "seconds\n")
print('\nBERT model training review example:',Reviews[0])
print('\nBERT model training sentiment example:',Target[0])

# Intialize the model
timer = time.time()
model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
# Compile the model with an appropriate optimizer, loss function, and metrics
optimizer = tf.keras.optimizers.Adam(learning_rate=2e-5)
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')
model.compile(optimizer=optimizer, loss=loss, metrics=[metric])
timer = time.time() - timer
print("\nTotal time initializing and compiling the BERT model:", timer, "seconds\n")

# Train the model
timer = time.time()
history = model.fit([X_train_encoded['input_ids'], X_train_encoded['token_type_ids'], X_train_encoded['attention_mask']], Target, batch_size=32, epochs=3)
timer = time.time() - timer
print("\nTotal time training the BERT model:", timer, "seconds\n")

# Evaluate the model on the test data
timer = time.time()
test_loss, test_accuracy = model.evaluate([X_test_encoded['input_ids'], X_test_encoded['token_type_ids'], X_test_encoded['attention_mask']], y_test)
print('\nBERT model test accuracy:', test_accuracy)
print('\nBERT model test loss:', test_loss)
timer = time.time() - timer
print("\nTotal time evaluating the BERT model:", timer, "seconds\n")
timer = time.time()
pred = model.predict([X_test_encoded['input_ids'], X_test_encoded['token_type_ids'], X_test_encoded['attention_mask']])
# pred is of type TFSequenceClassifierOutput
logits = pred.logits
# Use argmax along the appropriate axis to get the predicted labels
pred_labels = tf.argmax(logits, axis=1)
# Convert the predicted labels to a NumPy array
pred_labels = pred_labels.numpy()
label = {1: 'Positive',	0: 'Negative'}
timer = time.time() - timer
print("\nTotal time predicting with the BERT model:", timer, "seconds")

# Map the predicted labels to their corresponding strings using the label dictionary
pred_labels = [label[i] for i in pred_labels]
Actual = [label[i] for i in y_test]
print('\nBERT model predicted sentiments:', pred_labels[:10])
print('\nBERT model actual sentiments:', Actual[:10])
print('\nBERT model classification report:\n\n', classification_report(Actual, pred_labels))