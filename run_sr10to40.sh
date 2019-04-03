#!/bin/sh

#PBS -N neurosat_sr10t40
#PBS -l nodes=1:ppn=1:gpus=1
#PBS -o $PBS_JOBID.o
#PBS -e $PBS_JOBID.e
#PBS -d /home/zhangfan-mff/projects/neurosat/pytorch_neurosat/

python src/train.py \
  --task-name 'neurosat_4th_rnd' \
  --dim 128 \
  --n_rounds 26 \
  --epochs 200 \
  --n_pairs 100000 \
  --max_nodes_per_batch 12000 \
  --gen_log '/home/zhangfan-mff/projects/neurosat/pytorch_neurosat/log/data_maker_sr10t40.log' \
  --min_n 10 \
  --max_n 40 \
  --restore '/home/zhangfan-mff/projects/neurosat/pytorch_neurosat/model/neurosat_3rd_rnd_sr10to40_ep200_nr26_d128.pth.tar' \
  --val-file 'val_v40_vpb12000_b2604.pkl'
