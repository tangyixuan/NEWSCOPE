import os
import json
import torch

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--data_root_path", type=str)
args = parser.parse_args()
data_root_path = args.data_root_path

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# reranker model
from transformers import AutoModelForSequenceClassification, AutoTokenizer
reranker_model_name = 'bge-reranker-large'
reranker_tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-reranker-large')
reranker_model = AutoModelForSequenceClassification.from_pretrained('BAAI/bge-reranker-large').to(device)

# paths
emb_model_name = 'bilingual-embedding-large'
news_text_path = os.path.join(data_root_path, 'news_text_by_paragraph')
topic_text_fn = os.path.join(data_root_path, 'raw_data/headline.json')
annotation_path = os.path.join(data_root_path, "GPT-4o-mini_annotation")
reranker_score_path = os.path.join(data_root_path, f'reranker_score_{reranker_model_name}_on_positive_samples')


if not os.path.exists(reranker_score_path):
    os.makedirs(reranker_score_path)
os.chdir(reranker_score_path)

with open(topic_text_fn, 'r', encoding='utf-8') as f:
    topic_texts = json.load(f)

all_news_text = {}
for news_file_name in os.listdir(news_text_path):
    if not news_file_name.endswith("_by_paragraph.json") :
        continue
    f1 = open(os.path.join(news_text_path, news_file_name), 'r', encoding='utf-8')
    news_text = json.load(f1)
    all_news_text.update(news_text)

reranker_model.eval()
for news_file_name in os.listdir(annotation_path):
    print(news_file_name)
    if not news_file_name.endswith("_by_paragraph.json") :
        continue
    if os.path.isfile(news_file_name):
        continue
    
    topic_text = topic_texts[news_file_name.split("_")[0]]
    f1 = open(os.path.join(annotation_path, news_file_name), 'r', encoding='utf-8')
    positive_samples = json.load(f1)
    for _id in list(positive_samples.keys()):
        if positive_samples[_id] == 0:
            positive_samples.pop(_id)
    
    pairs = []
    rerank_scores = {}
    for para_id in positive_samples:
        story_text = all_news_text[para_id]
        pairs.append([topic_text, story_text])
    with torch.no_grad():
        input = reranker_tokenizer(pairs, padding=True, truncation=True, return_tensors='pt', max_length=512).to(device)
        scores = reranker_model(**input, return_dict=True).logits.view(-1, ).float()
        print(scores.shape)
        # sigmoid
        scores = torch.sigmoid(scores).cpu().numpy()
    idx = 0
    for para_id in positive_samples:
        rerank_scores[para_id] = float(scores[idx])
        idx += 1

    with open(news_file_name, 'w') as  f:
        json.dump(rerank_scores, f)