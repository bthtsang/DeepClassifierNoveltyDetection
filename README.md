# Neural network classifier and novelty detector for unevenly sampled light curves of variable stars
<![![DOI](https://zenodo.org/badge/90776775.svg)](https://zenodo.org/badge/latestdoi/90776775)!>

Code accompanying the paper "Deep Neural Network Classifier for Variable Stars with Novelty Detection Capability" (arXiv:).

The code for the recurrent neural network-based autoencoder was taken from the official implementation by [B. Naul et al. (2018)](https://github.com/bnaul/IrregularTimeSeriesAutoencoderPaper).
There is no publicly available implementation of the Deep Autoencoding Gaussian Mixture Model (DAGMM) by [B. Zong et al. (2018)](https://openreview.net/pdf?id=BJJLHbb0-). Our estimation network implementation, used for both classification and novelty detection, has benefited from the unofficial implementations by [danieltan07](https://github.com/danieltan07/dagmm), [Newcomer520](https://github.com/Newcomer520/tf-dagmm), and in particular [tnakae](https://github.com/tnakae/DAGMM). 

- Example slurm scripts to launch the joint and sequential training are in `train_joint.slurm` and `train_sequential.slurm`.
- Code for setting up the estimation network is in `estimation_network.py`.
- Code for the Gaussian Mixture model (GMM) is in `gmm.py`.
- For joint training, `survey_rnngmm_classifer.py` sets up and trains the RNN autoencoder and estimation network. In the case of sequential training, this script is responsible only for the RNN autoencoder training. 
- Classification accuracy, confusion matrix, sample energy histograms, and novelty detection scores are calculated in `classify_noveltydetect.py`. In the case of sequential training, the estimation network is set up and trained here. 

We adopt Naul's RNN autoencoder architecture, with changes made to the following to accommodate the additional estimation network: 
- Function for computing the reconstruction error features was added to `autoencoder.py`
- Additional hyperparameters for the estimation network were added to `keras_util.py`: 
  - `--gmm_on`: on switch for joint training, remove for sequential training.
  - `--lambda1`: coefficient lambda in front of the GMM loss `L_GMM` (Equation (6)). 
  - `--num_classes`: number of variable classes/GMM components.
  - `--estnet_size`: size(s) of the hidden layers excluding the input and output layers. For example, set `--estnet_size 16` for a single hidden layer with 16 units. 
  - `--estnet_drop_frac`: dropout rate of estimation network's hidden layer(s). Set `--estnet_drop_frac 0.0` to remove the dropout layer. 
- An example dataset of ASAS-SN light curves is included under `./data/asassn/`.
- Results and model weights are saved under `./keras_logs`.
