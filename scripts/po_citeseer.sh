export CUDA_VISIBLE_DEVICES=0
export CUDA_LAUNCH_BLOCKING=1
wandb offline
python main.py --conv-name gat \
    --mode po \
    --dataset citeseer \
    --po-epoch 10 \
    --po-sm-batch-size 16 \
    --po-batch-size 16 \
    --logging-steps 10 \
    --eval-steps 200 \
    --po-lr 1e-4 \
    --oracle-model-path '/root/autodl-tmp/FairLLM4Graph/checkpoints/citeseer/bert-base-uncased/save_model' \
    --ref-model-path '/root/autodl-tmp/FairLLM4Graph/checkpoints/citeseer/bert-base-uncased_filter/save_model' \
    --use-peft \
    --epoch 500 \
    # --filter \
    # --add-kl \
    
