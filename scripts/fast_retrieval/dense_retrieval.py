import os
import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import argparse
import time

parser = argparse.ArgumentParser()
parser.add_argument("--data_root_path", type=str)
parser.add_argument("--emb_model_name", type=str, default='bilingual-embedding-large')
parser.add_argument("--news_text_dir", type=str, default='news_text_by_paragraph')
parser.add_argument("--top_k", type=int , default=100)
args = parser.parse_args()
data_root_path = args.data_root_path
news_text_dir = args.news_text_dir
top_k = args.top_k
emb_model_name = args.emb_model_name

news_text_path = os.path.join(data_root_path, news_text_dir)
embedding_path = os.path.join(data_root_path, f'embedding_by_{emb_model_name}')
emb_sim_score_path = os.path.join(data_root_path, f'top_{top_k}_sim_score_all')

def compute_sim_score(embedding_path, output_sim_score_path):
    if not os.path.exists(output_sim_score_path):
        os.makedirs(output_sim_score_path)

    all_para_ids = []
    all_news_embddings = []
    start_time = time.time()
    for news_file_name in os.listdir(embedding_path):
        if not news_file_name.endswith("_by_paragraph.json") :
            continue
        print(news_file_name)
        f1 = open(os.path.join(embedding_path, news_file_name), 'r', encoding='utf-8')
        news_embedding = json.load(f1)

        for t in news_embedding:
            all_news_embddings.append(news_embedding[t])
            all_para_ids.append(t)

    print("length of all_embddings:", len(all_news_embddings))
    all_news_embddings = np.array(all_news_embddings)
    
    f = open(os.path.join(embedding_path, "topic_id_to_topic.json"), 'r', encoding='utf-8')
    topic_embeddings = json.load(f)

    all_topic_ids = []
    all_topic_embddings = []
    for topic_id in topic_embeddings:
        all_topic_ids.append(topic_id)
        all_topic_embddings.append(topic_embeddings[topic_id])
    all_topic_embddings = np.array(all_topic_embddings)
    if all_topic_embddings.ndim == 1:
        all_topic_embddings = all_topic_embddings.reshape(1, -1)
    
    scores = cosine_similarity(all_topic_embddings, all_news_embddings)
    print("scores.shape:", scores.shape)
    
    for t, topic_id in enumerate(all_topic_ids):
        scores_dict = {}
        for p, para_id in enumerate(all_para_ids):
            scores_dict[para_id] = scores[t][p].tolist()
        # sorted_result = sorted(scores_dict.keys(), key=lambda x: scores_dict[x], reverse=True)[:top_k]
        # output_dict = {k: scores_dict[k] for k in sorted_result}
        output_dict = scores_dict

        output_sim_score_fp = os.path.join(output_sim_score_path, f"{topic_id}_by_paragraph.json")
        with open(output_sim_score_fp, 'w') as  f:
            json.dump(output_dict, f)
    end_time = time.time()
    print("time cost:", end_time - start_time)
    print("average time cost:", (end_time - start_time) / len(all_topic_ids))
if __name__ == "__main__":
    compute_sim_score(embedding_path, emb_sim_score_path)