# Neural network classifier and novelty detector for unevenly sampled time series
<![![DOI](https://zenodo.org/badge/90776775.svg)](https://zenodo.org/badge/latestdoi/90776775)!>

Code accompanying the paper "Deep neural network classifier for periodic light curves with novelty detection" (arXiv:).

The code for the recurrent neural network-based autoencoder was taken from the [official implementation by B. Naul et al. (2018)](https://github.com/bnaul/IrregularTimeSeriesAutoencoderPaper).
There is no publicly available implementation of the Deep Autoencoding Gaussian Mixture Model (DAGMM) by [B. Zong et al. (2018)](https://openreview.net/pdf?id=BJJLHbb0-). The estimation network implementation here, used for both classification and novelty detection, has benefited from the unofficial implementation by [danieltan07](https://github.com/danieltan07/dagmm), [Newcomer520](https://github.com/Newcomer520/tf-dagmm), and in particular [tnakae](https://github.com/tnakae/DAGMM). 

- Code for setting up and training the RNN-autoencoder and estimation network is in `survey_autoencoder_gmm.py`
- Code for constructing the estimation network is in `estimation_network.py`.
- Code for constructing the Gaussian Mixture model is in `gmm.py`.
- Code for generating classification accuracy, confusion matrix, energy histograms and novelty detection results is found in `results.py`

We use Naul's RNN autoencoder architecture, with changes made to the following to accommodate the additional estimation network: 
- Autoencoder network architecture is defined in `autoencoder.py`
- Training parameters are defined in `keras_utils.py`
- Light curve data are stored under `./data/asassn/`
- Model weights are saved in `./keras_logs`

