import os 
import re
import nltk
import torch
import json
import argparse
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.tokenize import word_tokenize
from tqdm import tqdm
from keras.preprocessing.text import Tokenizer
from twitter.TwitterDataset import TwitterDataset

EMBEDDING_SIZE = 50
GLOVE_PATH = 'src/twitter/glove.6B.'+str(EMBEDDING_SIZE)+'d.txt'  # Assurez-vous que le fichier GloVe est présent

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_in_path', type=str,
                    help='CSV dataset path')
parser.add_argument('--dataset_out_path', type=str,
                    help='Path to save the processed dataset')
args = parser.parse_args()

if not args.dataset_out_path or not args.dataset_in_path:
    raise Exception('Please provide dataset paths')

if not os.path.exists(args.dataset_in_path):
    raise Exception('Dataset path does not exist')

nltk.download('stopwords')
nltk.download('punkt')  # Correction du nom du dataset
stop_words = set(stopwords.words('english'))
stemmer = SnowballStemmer('english')
text_cleaning_re = "@\S+|https?:\S+|http?:\S|[^A-Za-z0-9]+"

data = pd.read_csv(args.dataset_in_path, encoding='latin', header=None)
data.columns = ['sentiment', 'id', 'date', 'query', 'user_id', 'text']
data = data[data['sentiment'] != 2]
data['sentiment'] = data['sentiment'].map({0: 0, 4: 1})  # Convert sentiment labels
data = data.drop(['id', 'date', 'query', 'user_id'], axis=1)

#keep only half of the dataset for faster training (with the same distribution of labels)
# positive_data = data[data['sentiment'] == 1].sample(frac=0.5)
# negative_data = data[data['sentiment'] == 0].sample(frac=0.5)
# data = pd.concat([positive_data, negative_data])

def clean_text(text, stem=False):
    text = re.sub(text_cleaning_re, ' ', str(text).lower()).strip()
    tokens = [stemmer.stem(token) if stem else token for token in text.split() if token not in stop_words]
    return " ".join(tokens)

data.text = data.text.apply(lambda x: clean_text(x))

tokenized_text = data.text.apply(lambda x: word_tokenize(x))
labels = data.sentiment.tolist()
tokenizer = Tokenizer()
tokenizer.fit_on_texts(data.text)
word_index = tokenizer.word_index

# Charger les embeddings GloVe
glove_embeddings = {}
with open(GLOVE_PATH, 'r', encoding='utf-8') as f:
    for line in f:
        values = line.split()
        word = values[0]
        vector = np.asarray(values[1:], dtype='float32')
        glove_embeddings[word] = vector

vocab_size = len(word_index) + 1
embedding_matrix = np.zeros((vocab_size, EMBEDDING_SIZE))

for word, i in word_index.items():
    embedding_vector = glove_embeddings.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

embedded_sentences = []
masks = []
max_len = max(len(sentence) for sentence in tokenized_text) + 1 #adding 1 for the cls token

def sentence_to_embedding(sentence):
    return [glove_embeddings.get(word, np.zeros(EMBEDDING_SIZE)) for word in sentence]

for i in tqdm(range(len(tokenized_text))):
    sentence = tokenized_text[i] + ['cls']
    label = labels[i]
    embedded_sentence = sentence_to_embedding(sentence)
    mask = [0] * (max_len - len(embedded_sentence)) + [1]*len(embedded_sentence)
    embedded_sentence = [np.zeros(EMBEDDING_SIZE)] * (max_len - len(embedded_sentence)) + embedded_sentence
    #add the label at the end for the minGRU units
    embedded_sentences.append(embedded_sentence)
    masks.append(np.array(mask))

embedded_sentences = np.array(embedded_sentences)
masks = np.array(masks)

embedded_tweets = torch.tensor(embedded_sentences, dtype=torch.float)
masks_torch = torch.tensor(masks, dtype=torch.bool)

twitter_dataset = TwitterDataset(embedded_tweets, data['sentiment'].tolist(), masks_torch)
torch.save(twitter_dataset, args.dataset_out_path)
