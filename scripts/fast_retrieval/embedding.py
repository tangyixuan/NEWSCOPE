import os
import json
import torch
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--data_root_path", type=str)
parser.add_argument("--news_text_dir", type=str, default="news_text_by_paragraph")
args = parser.parse_args()
data_root_path = args.data_root_path

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# embedding model
from sentence_transformers import SentenceTransformer
emb_model_name = 'bilingual-embedding-large'
emb_model = SentenceTransformer('Lajavaness/bilingual-embedding-large', trust_remote_code=True, device=device)

news_text_path = os.path.join(data_root_path, args.news_text_dir)
embedding_path = os.path.join(data_root_path, f'embedding_by_{emb_model_name}')


def prepare_news_embedding(news_text_path, news_embedding_path, model):
    if not os.path.exists(news_embedding_path):
        os.makedirs(news_embedding_path)

    news_text_files = os.listdir(news_text_path)
    os.chdir(news_embedding_path)
    
    model.eval()
    for news_file_name in news_text_files:
        print(news_file_name)
        if news_file_name.split('_')[-1] != 'paragraph.json':
            continue
        news_text_file_path = os.path.join(news_text_path, news_file_name)
        f = open(news_text_file_path, 'r', encoding='utf-8')
        news_text = json.load(f)
        
        if os.path.isfile(news_file_name):
            continue

        batch_size = 100
        embeddings = []
        news_text_list = list(news_text.values())
        for i in range(0, len(news_text_list), batch_size):
            text = news_text_list[i:i+batch_size]
            with torch.no_grad():
                output = model.encode(text)
            embeddings.append(output)
        embeddings = np.concatenate(embeddings, axis=0)
        news_embedding = embeddings.tolist()

        news_embedding_dict = {}
        for i in range(len(news_embedding)):
            news_embedding_dict[list(news_text.keys())[i]] = news_embedding[i]
        with open(news_file_name, 'w', encoding='utf-8') as f:
            json.dump(news_embedding_dict, f)

    news_topic_file_path = os.path.join(data_root_path, "raw_data/headline.json")
    f = open(news_topic_file_path, 'r', encoding='utf-8')
    news_topic = json.load(f)
    topic_embedding = {}
    for topic_id in news_topic:
        if topic_id in topic_embedding:
            continue

        text = news_topic[topic_id]
        with torch.no_grad():
            output = model.encode(text)
        topic_embedding[topic_id] = output.tolist()
    news_topic_embd_path = os.path.join(news_embedding_path, "topic_id_to_topic.json")
    with open(news_topic_embd_path, 'w', encoding='utf-8') as f:
        json.dump(topic_embedding, f)

if __name__ == "__main__":
    prepare_news_embedding(news_text_path, embedding_path, emb_model)