export CUDA_VISIBLE_DEVICES=0
source /share/apps/anaconda3/2022.10/etc//profile.d/conda.sh
conda activate snnenv

if [ ! -d "$DIRECTORY" ]; then
    mkdir ../save
fi

model=vgg9_mini
dataset="MNIST"
date="4_17"
data_path="/share/seo/snn/pt/dvscifar10/T10/"

lr=0.001
lamb=0.90
batch_size=24
optimizer="adam"
lr_sch="cos"
T=4


log_file="training.log"
save_path="./save/${dataset}/${date}/Baseline/w32a32/T${T}/${model}/${model}_lr${lr}_batch${batch_size}_${loss}loss_run2/hp_mem"

#python -m torch.distributed.launch --nproc_per_node=1 --nnodes=1 --node_rank=0 ./main.py \
torchrun --nproc_per_node=1 --nnodes=1 main.py \
    --model ${model} \
    --lr ${lr} \
    --lamb ${lamb} \
    --lr_sch ${lr_sch} \
    --optimizer ${optimizer} \
    --batch-size ${batch_size} \
    --epochs 200 \
    --dataset ${dataset} \
    --T ${T} \
    --log_file ${log_file} \
    --save_path ${save_path} \
    --data_path ${data_path} \

