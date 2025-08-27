# Based on relevance re-ranking

import numpy as np
from sklearn.cluster import OPTICS
import hdbscan
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import torch
# from nltk.tokenize import sent_tokenize
import stanza
# stanza.download('en')
nlp = stanza.Pipeline(lang='en', processors='tokenize')
from collections import defaultdict
import os
import json
import argparse
from sentence_transformers import SentenceTransformer
max_top_k = 100

parser = argparse.ArgumentParser()
parser.add_argument("--data_root_path", type=str)
parser.add_argument("--w", type=float, help="w*diversity_score + similarity_score")
parser.add_argument("--news_text_dir", type=str, default="news_text_by_paragraph")
parser.add_argument("--greedy_select_result_dir", type=str, default="greedy_select")
parser.add_argument("--relevance_score_dir", type=str, default="reranker_score")
parser.add_argument("--sentences_cluster_dir", type=str)

args = parser.parse_args()
data_root_path = args.data_root_path
w = args.w

greedy_select_result_path = os.path.join(data_root_path, args.greedy_select_result_dir)
relevance_score_path = os.path.join(data_root_path, args.relevance_score_dir)
sentences_cluster_result = os.path.join(data_root_path, args.sentences_cluster_dir)
news_text_path = os.path.join(data_root_path, args.news_text_dir)

for path in [sentences_cluster_result, greedy_select_result_path]:
    if not os.path.exists(path):
        os.makedirs(path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SentenceTransformer('Lajavaness/bilingual-embedding-large', trust_remote_code=True, device=device)
# torch.set_default_tensor_type(torch.cuda.HalfTensor)  # Use FP16
all_news_text = {}
for news_file_name in os.listdir(news_text_path):
    if not news_file_name.endswith("_by_paragraph.json") :
        continue
    f1 = open(os.path.join(news_text_path, news_file_name), 'r', encoding='utf-8')
    news_text = json.load(f1)
    all_news_text.update(news_text)

def generate_sentence_embeddings(sentences, model):
    embeddings = []
    batch_size = 100
    # input the sentences by batch
    for i in range(0, len(sentences), batch_size):
        with torch.no_grad():
            output = model.encode(sentences[i:i+batch_size])
        embeddings.append(output)
    embeddings = np.concatenate(embeddings, axis=0)
    return embeddings

for news_file_name in os.listdir(relevance_score_path):
    if not news_file_name.endswith("_by_paragraph.json"):
        continue
    i = news_file_name.split("_")[0]
    print("******************************************")
    print(news_file_name)
    f1 = open(os.path.join(relevance_score_path, news_file_name), 'r', encoding='utf-8')
    relevance_score = json.load(f1)

    paragraphs = []
    idto_id = {}
    for _id in relevance_score:
        idto_id[len(paragraphs)] =  _id
        paragraphs.append(all_news_text[_id])
        
    print("# of paragraphs: ", len(paragraphs))

    sentences = []
    para_to_sent_mapping = defaultdict(list)
    sent_to_para_mapping = {}
    
    if os.path.exists(os.path.join(sentences_cluster_result, news_file_name)):
        with open(os.path.join(sentences_cluster_result, news_file_name), 'r') as f:
            cluster_result_dict = json.load(f)

        cluster_result = []
        for cluster_id in cluster_result_dict:
            for d in cluster_result_dict[cluster_id]:
                sent_id = len(sentences)
                sentences.append(d["text"])
                para_to_sent_mapping[d["paragraph id"]].append(sent_id)
                sent_to_para_mapping[sent_id] = d["paragraph id"]
                cluster_result.append(cluster_id)
    # if 1 == 0:
    #     pass
    else:
        # Segment paragraphs into sentences and create paragraph_id to sentence_id mapping
        for para_id, origin_para in enumerate(paragraphs):
            para = origin_para
            para_sentences = [sentence.text for sentence in nlp(para).sentences]
            # para_sentences = sent_tokenize(para)
            
            split_para_sentences = []
            for sent in para_sentences:
                if '\n' in sent:
                    for s in sent.split("\n"):
                        if len(s.split(" ")) > 5:
                            split_para_sentences += [s]
                elif len(sent.split(" ")) > 5:
                    split_para_sentences += [sent]

            for sent in split_para_sentences:
                if sent.strip():
                    sent_id = len(sentences)
                    sentences.append(sent)
                    para_to_sent_mapping[para_id].append(sent_id)
                    sent_to_para_mapping[sent_id] = para_id
        print("# of sentences:", len(sentences))

        # Generate sentence representations
        sentence_embeddings = generate_sentence_embeddings(sentences, model)
        tfidf = TfidfVectorizer(stop_words="english", max_df=0.9, max_features=len(sentences), ngram_range=(1, 2)).fit_transform(sentences).toarray()
        sentence_representations = np.concatenate((sentence_embeddings, tfidf), axis=1)
        print("sentence_representations shape:", sentence_representations.shape)

        cosine_sim_matrix = cosine_similarity(sentence_representations)
        distance_matrix = 1.0 - cosine_sim_matrix
        refined_distance_matrix = np.where(distance_matrix > 0, distance_matrix, 0) # fix error due to floating compuation
        
        ######################cluster by OPTICS######################
        clusterer = OPTICS(min_samples=2, metric='precomputed', n_jobs=-1)
        cluster_result = clusterer.fit_predict(refined_distance_matrix)
        # clusterer = hdbscan.HDBSCAN(metric='precomputed', min_samples = 1, gen_min_span_tree=True, cluster_selection_method='leaf', min_cluster_size = 2)
        # clusterer.fit(refined_distance_matrix)
        # Z = clusterer.single_linkage_tree_.to_numpy()
        # threshold = 0.4*max(Z[:,2])
        # cluster_result = fcluster(Z, threshold, criterion='distance')
        
        # Z = linkage(sentence_representations, method='average', metric='cosine')
        # threshold = 0.5*max(Z[:,2])
        # cluster_result = fcluster(Z, threshold, criterion='distance')


        # Generate cluster id to list of sentences in the cluster dict
        cluster_to_sentences = defaultdict(list)
        for sent_id, cluster_id in enumerate(cluster_result):
            cluster_to_sentences[cluster_id].append(sent_id)

        output_cluster_result = {}
        for cluster_id in cluster_to_sentences:
            output_cluster_result[int(cluster_id)] = []
            
            # print("*****************************************************")
            for sent_id in cluster_to_sentences[cluster_id]:
                # print(f"  {sentences[sent_id]} -- from paragraph {sent_to_para_mapping[sent_id]}")
                output_cluster_result[int(cluster_id)].append({
                    "text": sentences[sent_id], 
                    "paragraph id": sent_to_para_mapping[sent_id],
                    "paragraph _id": idto_id[sent_to_para_mapping[sent_id]],
                    "sentence num of this paragraph": len(para_to_sent_mapping[sent_to_para_mapping[sent_id]])
                    })
        with open(os.path.join(sentences_cluster_result, news_file_name), 'w', encoding='utf-8') as f:
            json.dump(output_cluster_result, f)

    # Initialize selected paragraphs set
    selected_paragraphs = set()
    covered_clusters = set()
    rank_by_sim_and_diversity = [max_top_k] * len(paragraphs)

    # Greedily select paragraphs
    cnt = 0
    while cnt < max_top_k:
        best_paragraph = None
        max_score = 0
        # Compute diversity score for each paragraph
        for para_id in range(len(paragraphs)):
            if para_id in selected_paragraphs:
                continue
            
            para_sent_ids = para_to_sent_mapping[para_id]
            para_clusters = set(cluster_result[sent_id] for sent_id in para_sent_ids)
            new_clusters = para_clusters - covered_clusters

            diversity_score = len(new_clusters)
            similarity_score = relevance_score[idto_id[para_id]]
            
            combined_score = w*diversity_score + similarity_score
            # print(diversity_score, similarity_score)
            if combined_score > max_score:
                max_score = combined_score
                best_paragraph = para_id
                best_cover_clusters = list(cluster_result[sent_id] for sent_id in para_sent_ids)
            elif combined_score == max_score:
                if len(para_sent_ids) < len(para_to_sent_mapping[best_paragraph]):
                    best_paragraph = para_id
                    best_cover_clusters = list(cluster_result[sent_id] for sent_id in para_sent_ids)

        if best_paragraph is None:
            break

        selected_paragraphs.add(best_paragraph)        
        covered_clusters.update(set(cluster_result[sent_id] for sent_id in para_to_sent_mapping[best_paragraph]))
        
        rank_by_sim_and_diversity[best_paragraph] = cnt
        cnt += 1

    # Print selected paragraphs and the clusters they cover
    # print("\nSelected paragraphs and their covered clusters:")
    for para_id in selected_paragraphs:
        para_sent_ids = para_to_sent_mapping[para_id]
        para_clusters = set(cluster_result[sent_id] for sent_id in para_sent_ids)
        # print(f"Paragraph {para_id}: Covers clusters {para_clusters}")

    # print(f"\nTotal clusters covered: {len(covered_clusters)} out of {len(total_clusters)}")
    # print(f"Coverage ratio: {len(covered_clusters) / len(total_clusters):.2f}")

    # store the greedy select ranking result
    output = {}
    idx = 0
    print(rank_by_sim_and_diversity)
    for _id in relevance_score:
        output[_id] = rank_by_sim_and_diversity[idx]
        idx += 1

    with open(os.path.join(greedy_select_result_path, news_file_name), 'w') as  f:
        json.dump(output, f)