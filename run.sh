DATA_ROOT=/home/lucliu/dataset
DATA=mscoco # mscoco, nus-wide 
DATA_PATH=${DATA_ROOT}/${DATA}

BASE_MODEL=baseline # ours, baseline, wildcat
EXTEND_MODEL=50 # 50, 101
MODEL=${BASE_MODEL}_${EXTEND_MODEL}

IMG_SIZE=448 # 448, 224 (default 448; if 224, change avgpooling 14 in baseline to 7)

# SAVE_ROOT=/opt/intern/users/lucliu/multilabel/checkpoints
SAVE_ROOT=./checkpoints
SAVE_PATH=${SAVE_ROOT}/${MODEL}/${DATA}/${IMG_SIZE}
CHECKPOINTS=${SAVE_PATH}/checkpoint.pth.tar

CUDA_VISIBLE_DEVICES=1,2 python3 -m demo ${DATA_PATH} \
--image-size ${IMG_SIZE} \
--batch-size 50 \
--lrp 0.1 \
--lr 0.01 \
--epochs 12 \
--k 0.2 \
--maps 8 \
--alpha 0.7 \
--dataname ${DATA} \
--model ${MODEL} \
--save ${SAVE_PATH} \
# --resume ${CHECKPOINTS}
