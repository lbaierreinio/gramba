import os 
import re
import nltk
import torch
import argparse
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from transformers import BertTokenizer
from imdb.IMDBDataset import IMDBDataset

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
nltk.download('punkt')  # Correction dataset name
stop_words = set(stopwords.words('english'))
stemmer = SnowballStemmer('english')
text_cleaning_re = "@\S+|https?:\S+|http?:\S|[^A-Za-z0-9]+"

data = pd.read_csv(args.dataset_in_path)
data['sentiment'] = data['sentiment'].map({"negative": 0, "positive": 1})  # Convert sentiment labels

def clean_text(text, stem=False):
    text = re.sub(text_cleaning_re, ' ', str(text).lower()).strip()
    tokens = [stemmer.stem(token) if stem else token for token in text.split() if token not in stop_words]
    return " ".join(tokens)

data.review = data.review.apply(lambda x: clean_text(x))

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
tokenizer.padding_side = 'left'

imdb_dataset = IMDBDataset(data.review.tolist(), data.sentiment.tolist(), tokenizer)
torch.save(imdb_dataset, args.dataset_out_path)