DATA_ROOT=/home/lucliu/dataset
DATA=voc # mscoco, nus-wide 
DATA_PATH=${DATA_ROOT}/${DATA}

BASE_MODEL=ours # ours, baseline, wildcat, map_attn(dev), noise(dev), no_attn(dev), feat_attn(dev)
EXTEND_MODEL=50 # 50, 101
MODEL=${BASE_MODEL}_${EXTEND_MODEL}

IMG_SIZE=224 # 448, 224 (default 448; if 224, change avgpooling 14 in baseline to 7)

SAVE_ROOT=/opt/intern/users/lucliu/multilabel/checkpoints/dev
# SAVE_ROOT=./checkpoints
SAVE_PATH=${SAVE_ROOT}/${MODEL}/${DATA}/${IMG_SIZE}
CHECKPOINTS=${SAVE_PATH}/checkpoint.pth.tar

CUDA_VISIBLE_DEVICES=4 python3 -m demo ${DATA_PATH} \
--image-size ${IMG_SIZE} \
--batch-size 50 \
--lrp 0.1 \
--lr 0.01 \
--epochs 20 \
--k 0.2 \
--maps 8 \
--alpha 0.7 \
--dataname ${DATA} \
--model ${MODEL} \
--save ${SAVE_PATH} \
# --resume ${CHECKPOINTS}
