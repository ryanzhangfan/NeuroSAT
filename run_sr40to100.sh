#!/bin/sh

#PBS -N neurosat_sr40t100
#PBS -l nodes=1:ppn=1:gpus=1
#PBS -o $PBS_JOBID.o
#PBS -e $PBS_JOBID.e
#PBS -d /home/zhangfan-mff/projects/neurosat/pytorch_neurosat/

python src/train.py \
  --task-name 'neurosat_3rd_rnd' \
  --dim 128 \
  --n_rounds 32 \
  --epochs 200 \
  --n_pairs 100000 \
  --max_nodes_per_batch 12000 \
  --gen_log '/home/zhangfan-mff/projects/neurosat/pytorch_neurosat/log/data_maker_sr40t100.log' \
  --min_n 40 \
  --max_n 100 \
  --restore '/home/zhangfan-mff/projects/neurosat/pytorch_neurosat/model/neurosat_2nd_rnd_sr40to100_ep200_nr32_d128_last.pth.tar' \
  --val-file 'val_v100_vpb12000_b1284.pkl'
