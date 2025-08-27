
set -e

data_root_path=$1
diverse_rerank_method=$2
code_root_path="./scripts"

# export CUDA_VISIBLE_DEVICES=1

############################################################################################################
# Step 1: Do dense retrieval.
echo "dense retrieval"
python $code_root_path/fast_retrieval/embedding.py\
        --data_root_path $data_root_path
python $code_root_path/fast_retrieval/dense_retrieval.py\
        --data_root_path $data_root_path  --emb_model_name bilingual-embedding-large

############################################################################################################
# Step 2: Get relevance score from re-ranker.
echo "relevance re-rank"
python $code_root_path/relevance_and_diverse_rerank/Relevance_Rerank/rerank_top_paragraphs.py\
        --data_root_path $data_root_path

############################################################################################################
# Step 3: Diverse re-rank.
case "$diverse_rerank_method" in
"GreedySCS")
echo "Method: Greedy Cluster Selection (GreedySCS)"
python $code_root_path/relevance_and_diverse_rerank/Greedy_Sentence_Cluster_Select/greedy_select_weighted_sum.py\
        --data_root_path $data_root_path --w 0.04 --sentences_cluster_dir "OPTICS_sent_cluster"
;;
"GreedyPlus")
echo "Method: Cluster-Based Weighting (GreedyPlus)"
python $code_root_path/relevance_and_diverse_rerank/Greedy_Plus/cluster_score_enhanced_greedy_select.py\
        --data_root_path $data_root_path --w_cluster 2 --w_sim 1
;;
*)
echo "Unknown diverse_rerank_method: $diverse_rerank_method. Expected 'GreedySCS' or 'GreedyPlus'."
exit 1
;;
esac