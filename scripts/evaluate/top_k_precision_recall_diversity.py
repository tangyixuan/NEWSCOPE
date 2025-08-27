import json
import os
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import sent_tokenize
# import stanza
# stanza.download('en')
# nlp = stanza.Pipeline(lang='en', processors='tokenize')
import argparse
import csv

parser = argparse.ArgumentParser()
parser.add_argument("--data_root_path", type=str)
parser.add_argument("--result_fn", type=str)
args = parser.parse_args()
data_root_path = args.data_root_path
result_fn = args.result_fn

emb_model_name = 'bilingual-embedding-large'
embedding_path = os.path.join(data_root_path, f'embedding_by_{emb_model_name}')
emb_sim_score_path = os.path.join(data_root_path, "top_100_sim_score")

reranker_model_name = 'bge-reranker-large'
reranker_score_path = os.path.join(data_root_path, f'reranker_score')

annotation_path = os.path.join(data_root_path, 'GPT-4o-mini_annotation')
dr_sent_cluster_path = os.path.join(data_root_path, 'OPTICS_sent_cluster')
pos_sent_cluster_path = os.path.join(data_root_path, 'sentences_clusters_on_positive_samples')
news_text_path = os.path.join(data_root_path, 'news_text_by_paragraph')

greedy_select_result_path = os.path.join(data_root_path, 'greedy_select')
cluster_score_based_greedy_select_result_path = os.path.join(data_root_path, 'cluster_score_based_greedy_select_cluster_2_sim_1')


all_para_embddings = {}
for news_file_name in os.listdir(embedding_path):
    if not news_file_name.endswith("_by_paragraph.json") :
        continue
    f1 = open(os.path.join(embedding_path, news_file_name), 'r', encoding='utf-8')
    news_embedding = json.load(f1)
    all_para_embddings.update(news_embedding)

all_news_text = {}
for news_file_name in os.listdir(news_text_path):
    if not news_file_name.endswith("_by_paragraph.json") :
        continue
    f1 = open(os.path.join(news_text_path, news_file_name), 'r', encoding='utf-8')
    news_text = json.load(f1)
    all_news_text.update(news_text)

def compute_metrics(score_or_rank, score_or_rank_path, annotation_path, top_k, all_para_embddings, method_name):
    avg_precision, avg_recall, avg_diversity, avg_cluster_sentence_rate, avg_pos_cluster_coverage, F1 = 0, 0, 0, 0, 0, 0
    if score_or_rank == "score":
        _reverse = True
    else:
        _reverse = False

    file_cnt = 0
    for news_file_name in os.listdir(annotation_path):
        if not news_file_name.endswith("_by_paragraph.json"):
            continue
        file_cnt += 1
        f = open(os.path.join(score_or_rank_path, news_file_name), 'r', encoding='utf-8')
        retrieve_result = json.load(f)
        f = open(os.path.join(annotation_path, news_file_name), 'r', encoding='utf-8')
        annotation_dict = json.load(f)
        f = open(os.path.join(dr_sent_cluster_path, news_file_name), 'r', encoding='utf-8')
        dr_sent_cluster = json.load(f)
        f = open(os.path.join(pos_sent_cluster_path, news_file_name), 'r', encoding='utf-8')
        pos_sent_cluster = json.load(f)
        
        embeddings_filtered = []
        paragraphs = []
        sorted_result = sorted(retrieve_result.keys(), key=lambda x: retrieve_result[x], reverse=_reverse)[:top_k]
        for _id in sorted_result:
            embeddings_filtered.append(all_para_embddings[_id])
            paragraphs.append(all_news_text[_id])

        cnt, TP, FP = 0, 0, 0
        for _id in sorted_result:
            cnt += 1
            if _id in annotation_dict:
                TP += 1
            else:
                FP += 1
            if cnt == top_k:
                break
        
        precision = TP / top_k
        recall = TP / len(annotation_dict)
        avg_precision += precision
        avg_recall += recall

        total = 0
        cnt = 0
        for m, embedding1 in enumerate(embeddings_filtered):
            for n in range(m+1, len(embeddings_filtered)):
                cnt += 1
                embedding2 = embeddings_filtered[n]
                total = total + 1 - cosine_similarity([embedding1], [embedding2])[0][0]
        diversity = total / cnt
        avg_diversity += diversity

        sentences = []
        for para_id, origin_para in enumerate(paragraphs):
            para = origin_para
            para_sentences = sent_tokenize(para)
            for sent in para_sentences:
                if '\n' in sent:
                    for s in sent.split("\n"):
                        if len(s.split(" ")) > 5:
                            sentences += [s]
                elif len(sent.split(" ")) > 5:
                    sentences += [sent]

        cluster_sentence_rate = 0
        num_sentences = len(sentences)
        for cluster_id in dr_sent_cluster:
            cnt_sent = 0
            for s in dr_sent_cluster[cluster_id]:
                if s["text"] in sentences:
                    cnt_sent += 1
            if cnt_sent != 0:
                cluster_sentence_rate += 1
        cluster_sentence_rate /= num_sentences
        avg_cluster_sentence_rate += cluster_sentence_rate

        pos_cluster_coverage = 0
        for cluster_id in pos_sent_cluster:
            for s in pos_sent_cluster[cluster_id]:
                if s["text"] in sentences:
                    pos_cluster_coverage += 1
                    break
        # print("num of positive clusters:", len(pos_sent_cluster))
        # print("num of positive clusters covered:", pos_cluster_coverage)
        pos_cluster_coverage /= len(pos_sent_cluster)
        avg_pos_cluster_coverage += pos_cluster_coverage

    avg_precision /= file_cnt
    avg_recall /= file_cnt
    F1 = 2*avg_precision*avg_recall / (avg_precision+avg_recall)
    avg_diversity /= file_cnt
    avg_cluster_sentence_rate /= file_cnt
    avg_pos_cluster_coverage /= file_cnt

    print("P:", avg_precision)
    print("R:", avg_recall)
    print("F1:", F1)
    print("D:", avg_diversity)
    print("I:", avg_cluster_sentence_rate)
    print("C:", avg_pos_cluster_coverage)

    return avg_precision, avg_recall, F1, avg_diversity, avg_cluster_sentence_rate, avg_pos_cluster_coverage

if __name__ == '__main__':
    output = {}
    for top_k in [5, 10, 20, 50]:
        print(f"Top_{top_k}")
        output[top_k] = {}

        # Dense_retrieval
        print(f"Top_{top_k}  dense_retrieval")
        avg_precision, avg_recall, F1, avg_diversity, avg_cluster_sentence_rate, avg_pos_cluster_coverage = compute_metrics("score", emb_sim_score_path, annotation_path, top_k, all_para_embddings, "DenseRetr")
        output[top_k]["DenseRetr"] = {"precision": avg_precision, "recall": avg_recall, "F1": F1, "diversity": avg_diversity, "cluster_sentence_rate": avg_cluster_sentence_rate, "pos_cluster_coverage": avg_pos_cluster_coverage}

        # Greedy_sentence_select
        print(f"Top_{top_k}  Greedy_sentence_select")
        avg_precision, avg_recall, F1, avg_diversity, avg_cluster_sentence_rate, avg_pos_cluster_coverage = compute_metrics("rank", greedy_select_result_path, annotation_path, top_k, all_para_embddings, "GreedySCS")
        output[top_k]["GreedySCS"] = {"precision": avg_precision, "recall": avg_recall, "F1": F1, "diversity": avg_diversity, "cluster_sentence_rate": avg_cluster_sentence_rate, "pos_cluster_coverage": avg_pos_cluster_coverage}

        # Cluster_score_based Greedy_sentence_select
        print(f"Top_{top_k}  cluster_score_based_greedy_select")
        avg_precision, avg_recall, F1, avg_diversity, avg_cluster_sentence_rate, avg_pos_cluster_coverage = compute_metrics("rank", cluster_score_based_greedy_select_result_path, annotation_path, top_k, all_para_embddings, "GreedyPlus")
        output[top_k]["GreedyPlus"] = {"precision": avg_precision, "recall": avg_recall, "F1": F1, "diversity": avg_diversity, "pos_cluster_coverage": avg_pos_cluster_coverage, "cluster_sentence_rate": avg_cluster_sentence_rate}

    if not os.path.exists(os.path.join(data_root_path, "eval_result")):
        os.makedirs(os.path.join(data_root_path, "eval_result"))
        
    with open(os.path.join(data_root_path, "eval_result", result_fn), 'w', encoding='utf-8') as f:
        writer = csv.writer(f)
        to_write_data = ['method']
        for k in output:
            for metric_name in ["precision", "recall", "F1", "diversity", "pos_cluster_coverage", "cluster_sentence_rate"]:
                to_write_data.append(f"top_{k}_{metric_name}")

        writer.writerow(to_write_data)

        for method_name in output[5]:
            to_write_data = [method_name]
            for k in output:
                for metric_name in ["precision", "recall", "F1", "diversity", "pos_cluster_coverage", "cluster_sentence_rate"]:
                    to_write_data.append(output[k][method_name][metric_name])
            writer.writerow(to_write_data)