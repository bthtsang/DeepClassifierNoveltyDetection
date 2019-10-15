# Neural network classifier and novelty detector for unevenly sampled light curves of variable stars
[![DOI](https://zenodo.org/badge/DOI/10.3847/2041-8213/ab212c.svg)](https://doi.org/10.3847/2041-8213/ab212c)

This is the official implementation accompanying the paper "Deep Neural Network Classifier for Variable Stars with Novelty Detection Capability" (arXiv: https://arxiv.org/abs/1905.05767).
Our motivation was to explore the application of deep neural networks for more than one single task beyond variable classification. 

The code for the recurrent neural network-based autoencoder was taken from the official implementation by B. Naul et al (2018) \[[View Code](https://github.com/bnaul/IrregularTimeSeriesAutoencoderPaper), [Read Paper](https://arxiv.org/abs/1711.10609)\].
There is no publicly available official implementation of the Deep Autoencoding Gaussian Mixture Model (DAGMM) by B. Zong et al. (2018) \[[Read Paper](https://openreview.net/pdf?id=BJJLHbb0-)\]. Our estimation network and Gaussian Mixture Model (GMM) implementation have benefited tremendously from the unofficial implementations by [danieltan07](https://github.com/danieltan07/dagmm), [Newcomer520](https://github.com/Newcomer520/tf-dagmm), and in particular [tnakae](https://github.com/tnakae/DAGMM). 

## Requirements
We have trained the models with the following packages:
- numpy (1.14.3)
- Tensorflow (1.9.0, GPU version)
- Keras (2.1.6)
- joblib (0.12.2)
- sklearn (0.19.1)

## Dataset
We have made extensive use of the All-Sky Automated Survey for Supernova (ASAS-SN) Variable Stars Database \[[Visit database](https://asas-sn.osu.edu/variables), [Read Paper](https://arxiv.org/abs/1809.07329)\].
An example dataset of ASAS-SN variable star light curves is included under `./data/asassn/sample.pkl`.

## File descriptions
- The RNN autoencoder network is defined in `autoencoder.py`, which is adopted from B. Naul's implementation. 
  We have added functions for computing the reconstruction error features. 
- The estimation network is defined in `estimation_network.py`.
- Code for setting up and fitting the GMM components is in `gmm.py`. In particular, the function to compute sample energy `gmm.energy` is defined here.
- For joint training, `survey_rnngmm_classifer.py` sets up and trains the RNN autoencoder and estimation network. In the case of sequential training, this script is responsible only for the RNN autoencoder training. 
- Classification accuracy, confusion matrix, sample energy histograms, and novelty detection scores are calculated in `classify_noveltydetect.py`. In the case of sequential training, the estimation network is set up and trained here, followed by the calculation of results. 


## How to train?
Example slurm scripts to launch the joint and sequential training can be found in `train_joint.slurm` and `train_sequential.slurm`.
On a local machine, joint training can be launched by:
```console
$ python survey_rnngmm_classifier.py --batch_size 500 --nb_epoch 250 --model_type gru --size 96 --num_layers 2 --embedding 16 --period_fold --drop_frac 0.25 --gmm_on --estnet_size 16 --num_classes 8 --estnet_drop_frac 0.5 --lambda1 0.001 --lr 2.0e-4 --ss_resid 0.7 --class_prob 0.9 --sim_type asassn --survey_files data/asassn/sample.pkl
```

## Descriptions of runtime/training parameters
At first sight the above command may look overwhelming, here's a list explaining what each parameter controls.
- Input/Output parameters:
  - `--sim_type`: name of output directory, to be appended after `./keras_logs/`. In the above example, training results and neural network weights will be stored under `./keras_logs/asassn/`. 
  - `--survey_files`: location of the pkl file containing input light curves. Example code for converting raw light curves into a single pkl file can be found in `light_curve.py`.
- Autoencoder parameters:
  - `--batch_size`: size of the minibatch (number of light curve sequences) used in training. 
  - `--nb_epoch`: number of epochs to train the joint network for; in the case of sequential training, this controls the number of epochs for autoencoder training.
  - `--model_type`: type of RNN layer, available options are `gru` (Gated Recurrent Unit), `lstm` (Long Short-term memory), and `vanilla` (simple RNN structure).
  - `--size`: number of unit in each RNN layer. 
  - `--num_layers`: number of RNN layers.
  - `--embedding`: embedding size for the autoencoder.
  - `--n_min` and `--n_max`: minimum and maximum number of observational epochs per light curve sequence. We set both to be 200 in the current work.
  - `--period_fold`: whether to phase-fold input light curves. Remove this flag for training without period-folding the light curves. 
  - `--drop_frac`: dropout rate of the RNN layers. Set `--drop_frac 0.0` to disable the dropout layer.
- Estimation network parameters:
  - `--gmm_on`: on switch for joint training, remove this flag for sequential training.
  - `--lambda1`: coefficient lambda in front of the GMM loss `L_GMM` (Equation (6)). 
  - `--estnet_size`: size(s) of the hidden layers excluding the input and output layers. For example, set `--estnet_size 16` for a single hidden layer with 16 units. 
  - `--estnet_drop_frac`: dropout rate of estimation network's hidden layer(s). Set `--estnet_drop_frac 0.0` to disable the dropout layer. 
  - `--num_classes`: number of variable classes/GMM components.
  - `--cls_epoch`: number of epochs to train the estimation network for, used only for sequential training. 
- Additional training parameters:
  - `--lr`: learning rate for training.
  - `--ss_resid`: (optional) maximum threshold of super-smoother residual for filtering input light curves. 
  - `--class_prob`: (optional) minimum ASAS-SN classification probability for filtering input light curves. 
