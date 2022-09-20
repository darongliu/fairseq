TASK_DATA=/home/darong/frequent_data/wav2vecu/timit_processed/matched/feat/precompute_pca512_cls128_mean_pooled
CPT_PATH=/home/darong/darong/fairseq/examples/wav2vec/unsupervised/multirun/2022-09-19/21-37-23/0/checkpoint_best.pt
SAVE_PATH=/home/darong/frequent_data/wav2vecu/results

python w2vu_generate.py --config-dir config/generate --config-name viterbi \
fairseq.common.user_dir=${FAIRSEQ_ROOT}/examples/wav2vec/unsupervised \
fairseq.task.data=${TASK_DATA} \
fairseq.common_eval.path=${CPT_PATH} \
fairseq.dataset.gen_subset=valid \
results_path=${SAVE_PATH}