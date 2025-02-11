import os 
import re
import nltk
import torch
import argparse
import pandas as pd
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
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
stop_words = set(stopwords.words('english'))

data = pd.read_csv(args.dataset_in_path)
data['sentiment'] = data['sentiment'].map({'positive': 1, 'negative': 0})

def clean_text(text):
    # Remove HTML tags
    text = BeautifulSoup(text, "html.parser").get_text()
    
    # Remove punctuation and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove stopwords
    text = ' '.join([word for word in text.split() if word not in stop_words])
    
    return text

# Apply function to the dataframe
data['cleaned_review'] = data['review'].apply(clean_text)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

imdb_dataset = IMDBDataset(data['cleaned_review'].tolist(), data['sentiment'].tolist(), tokenizer)

torch.save(imdb_dataset, args.dataset_out_path)