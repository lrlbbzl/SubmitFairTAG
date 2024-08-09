export CUDA_VISIBLE_DEVICES=0
export CUDA_LAUNCH_BLOCKING=1
wandb offline
python main.py --conv-name gcn \
    --mode gnn \
    --dataset cora \
    --epoch 500 \
    # --use-peft \
    --lm-epochs 10 \
    --lm-batch-size 32 \
    --sm-batch-size 16 \
    --oracle-sm-batch-size 16\
    --oracle-batch-size 32\
    --logging-steps 10 \
    --eval-steps 200 \
    --ft-lr 1e-3 \
    # --filter \
    # --add-kl
    
