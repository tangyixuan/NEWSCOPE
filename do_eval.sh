
set -e

data_root_path=$1
output_fn=$2
code_root_path="./scripts"

# export CUDA_VISIBLE_DEVICES=1

############################################################################################################
# Step 1: Prepare positive clusters for computing diversity metrics.
echo "compute sentence clusters on positive samples"
python $code_root_path/preprocess/sent_cluster_on_pos_samples.py\
        --data_root_path $data_root_path

############################################################################################################
# Step 2: Evaluation
echo "Evaluating......"
python $code_root_path/evaluate/top_k_precision_recall_diversity.py\
        --data_root_path $data_root_path --output_fn $output_fn