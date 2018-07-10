CUDA_VISIBLE_DEVICES=1,2 python3 -m demo_nus /home/lucliu/dataset/nus-wide \
--image-size 448 \
--batch-size 50 \
--lrp 0.1 \
--lr 0.01 \
--epochs 12 \
--k 0.2 \
--maps 8 \
--alpha 0.7 \