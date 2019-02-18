#!/bin/bash
#SBATCH -J train-emb16
#SBATCH -o train-emb16.o%j
#SBATCH -N 1 --partition=gpu --ntasks-per-node=1
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --nodes=1

source ~/.bashrc 

/bin/hostname
srun --gres=gpu:1 /usr/bin/nvidia-smi 

arg_string="--batch_size 500 --nb_epoch 250 --size 96 --num_layers 2 --embedding 16 --estnet_size 128 8 --num_classes 8 --period_fold --drop_frac 0.25 --estnet_drop_frac 0.5 --lambda1 0.1 --lambda2 1.0e-10 --lr 2.0e-4 --ss_resid 0.7 --model_type gru --gmm_on --sim_type asassn --survey_files data/asassn/asassn.pkl"

python survey_rnngmm_classifier.py ${arg_string[@]}
python nn_classifier_mixed_f64_cp_nocov.py ${arg_string[@]}


