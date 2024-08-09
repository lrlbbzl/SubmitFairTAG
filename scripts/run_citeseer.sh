export CUDA_VISIBLE_DEVICES=0
export CUDA_LAUNCH_BLOCKING=1
wandb offline
python main.py --conv-name gat \
    --mode ft_lm \
    --dataset citeseer \
    --epoch 500 \
    --lm-epochs 10 \
    --lm-batch-size 32 \
    --sm-batch-size 32 \
    --oracle-sm-batch-size 32 \
    --oracle-batch-size 32 \
    --logging-steps 10 \
    --eval-steps 500 \
    --oracle-model-path '/root/autodl-tmp/FairLLM4Graph/checkpoints/citeseer/bert-base-uncased/save_model' \
    --ref-model-path '/root/autodl-tmp/FairLLM4Graph/checkpoints/citeseer/bert-base-uncased_filter/save_model' \
    --ft-lr 1e-3 \
    --use-peft \
    # --filter \
    # --add-kl
    
