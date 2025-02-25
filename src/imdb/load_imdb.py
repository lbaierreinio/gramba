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
from tqdm import tqdm
from keras.preprocessing.text import Tokenizer
from imdb.IMDBDataset import IMDBDataset

EMBEDDING_SIZE = 300
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

data = pd.read_csv(args.dataset_in_path)
data['sentiment'] = data['sentiment'].map({"negative": 0, "positive": 1})  # Convert sentiment labels
print(data.head())
print(data['sentiment'].value_counts())

#keep only half of the dataset for faster training (with the same distribution of labels)
# positive_data = data[data['sentiment'] == 1].sample(frac=0.5)
# negative_data = data[data['sentiment'] == 0].sample(frac=0.5)
# data = pd.concat([positive_data, negative_data])

print("Cleaning text")
def clean_text(text, stem=False):
    text = re.sub(text_cleaning_re, ' ', str(text).lower()).strip()
    tokens = [stemmer.stem(token) if stem else token for token in text.split() if token not in stop_words]
    return " ".join(tokens)

data.review = data.review.apply(lambda x: clean_text(x))
print(data.head())

print("Tokenizing text")
tokenizer = Tokenizer()
tokenizer.fit_on_texts(data.review)
tokenizer.word_index['cls'] = len(tokenizer.word_index) + 1

word_index = tokenizer.word_index
#save word index
with open('src/imdb/word_index.json', 'w') as f:
    json.dump(word_index, f)
vocab_size = len(tokenizer.word_index) + 1
print("Vocabulary Size :", vocab_size)
#save vocab size
with open('src/imdb/vocab_size.txt', 'w') as f:
    f.write(str(vocab_size))
tokenized_text = tokenizer.texts_to_sequences(data.review)
labels = data.sentiment.tolist()

# Load GloVe embeddings
print("Loading GloVe embeddings")
embeddings_index = {}

f = open(GLOVE_PATH)
for line in f:
  values = line.split()
  word = value = values[0]
  coefs = np.asarray(values[1:], dtype='float32')
  embeddings_index[word] = coefs
f.close()
print ('Found %s word vectors.' % len(embeddings_index))
embedding_matrix = np.zeros((vocab_size, EMBEDDING_SIZE))
for word, i in word_index.items():
  embedding_vector = embeddings_index.get(word)
  if embedding_vector is not None:
    embedding_matrix[i] = embedding_vector

#save embedding matrix
np.save('src/imdb/embedding_matrix.npy', embedding_matrix)

tokenized_padded_text = []
masks = []
max_len = max(len(sentence) for sentence in tokenized_text) + 1 #adding 1 for the cls token

cls_token = word_index['cls']
padding_token = 0
print("Preparing pad and masks")
for i in tqdm(range(len(tokenized_text))):
    sentence = tokenized_text[i] + [cls_token]
    label = labels[i]
    mask = [padding_token] * (max_len - len(sentence)) + [1]*len(sentence)
    tokenized_padded_sentence = [padding_token] * (max_len - len(sentence)) + sentence
    #add the label at the end for the minGRU units
    tokenized_padded_text.append(tokenized_padded_sentence)
    masks.append(np.array(mask))

tokenized_padded_text = np.array(tokenized_padded_text)
masks = np.array(masks)

tokenized_padded_review = torch.tensor(tokenized_padded_text, dtype=torch.long)
masks_torch = torch.tensor(masks, dtype=torch.bool)

IMDB_dataset = IMDBDataset(tokenized_padded_review, labels, masks_torch)
torch.save(IMDB_dataset, args.dataset_out_path)
