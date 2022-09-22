PREFIX=w2v_unsup_gan_xp

# For wav2vec-U, audio features are pre-segmented
CONFIG_NAME=w2vu
TASK_DATA=/home/darong/frequent_data/wav2vecu/timit_processed/matched/feat/precompute_pca512_cls128_mean_pooled

# Unpaired text input
TEXT_DATA=/home/darong/frequent_data/wav2vecu/timit_processed/matched/phones  # path to fairseq-preprocessed GAN data (phones dir)
KENLM_PATH=/home/darong/frequent_data/wav2vecu/timit_processed/matched/phones/train_text_phn.04.bin  # KenLM 4-gram phoneme language model (LM data = GAN data here)

# PYTHONPATH=$FAIRSEQ_ROOT PREFIX=$PREFIX fairseq-hydra-train \
#     -m --config-dir config/gan \
#     --config-name $CONFIG_NAME \
#     task.data=${TASK_DATA} \
#     task.text_data=${TEXT_DATA} \
#     task.kenlm_path=${KENLM_PATH} \
#     common.user_dir=${FAIRSEQ_ROOT}/examples/wav2vec/unsupervised \
#     model.code_penalty=2,4 model.gradient_penalty=1.5,2.0 \
#     model.smoothness_weight=0.5,0.75,1.0 'common.seed=range(0,5)'

EXP_NAME=test

PYTHONPATH=$FAIRSEQ_ROOT PREFIX=$PREFIX fairseq-hydra-train \
    -m --config-dir config/gan \
    --config-name $CONFIG_NAME \
    hydra.sweep.dir=multirun/${EXP_NAME} \
    task.data=${TASK_DATA} \
    task.text_data=${TEXT_DATA} \
    task.kenlm_path=${KENLM_PATH} \
    common.user_dir=${FAIRSEQ_ROOT}/examples/wav2vec/unsupervised \
    model.code_penalty=2 model.gradient_penalty=1.5 \
    model.smoothness_weight=0.5 common.seed=0 \
    optimization.max_update=100000 