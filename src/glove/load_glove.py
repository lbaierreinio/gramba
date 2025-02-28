import argparse
import numpy as np
from transformers import BertTokenizer

parser = argparse.ArgumentParser()

parser.add_argument('--embedding_dim', type=int,
                    help='Embedding_dim')
parser.add_argument('--glove_path', type=str,
                    help='Path to GloVe embeddings of the desired size')
parser.add_argument('--output_path', type=str,
                    help='Path to save the embedding matrix')


args = parser.parse_args()

EMBEDDING_SIZE = args.embedding_dim
GLOVE_PATH = args.glove_path
OUTPUT_PATH = args.output_path

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
word_index = tokenizer.get_vocab()
vocab_size = len(word_index)

embeddings_index = {}

f = open(GLOVE_PATH)
for line in f:
  values = line.split()
  word = values[0]
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
np.save(OUTPUT_PATH, embedding_matrix)
