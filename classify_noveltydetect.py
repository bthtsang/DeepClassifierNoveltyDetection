import numpy as np
import keras.backend as K
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import joblib
from light_curve import LightCurve # For using Naul's LightCurve class
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import StratifiedKFold
from survey_rnngmm_classifier import main as survey_autoencoder
from survey_rnngmm_classifier import preprocess, energy_loss
from keras.models import Model
from keras_util import parse_model_args, get_run_id

# For one-hot vector conversion
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils

### For GMM training
from gmm import GMM
from autoencoder import extract_features
from estimation_net import EstimationNet
from keras.layers import Input, Lambda
from keras.optimizers import Adam
import keras_util as ku

### For novelty detection scores
from sklearn.metrics import precision_recall_fscore_support

### For RF classifier
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

# For generating confusion matrix
import itertools
from sklearn.metrics import confusion_matrix
import seaborn as sns
sns.set_context('paper', font_scale=1.6)

SEED = 0

def plot_confusion_matrix(y, y_pred, classnames, filename=None):
    classnames = sorted(classnames)
    sns.set_style("whitegrid", {'axes.grid' : False})
    cm = confusion_matrix(y, y_pred, classnames)
    valid_num = np.trace(cm)
    total_num = np.sum(cm)
    print ("Validation Accuracy", valid_num/total_num)
    cm = cm.astype('float') / cm.sum(axis=1)[:,np.newaxis]
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues, vmin=0,
               vmax=1.0)
    plt.colorbar()
    tick_marks = np.arange(len(classnames))
    plt.xticks(tick_marks, classnames, rotation=45)
    plt.yticks(tick_marks, classnames)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, "{0:.2f}".format(cm[i, j]),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black",
                 fontsize=9)

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label');
    if filename:
        plt.savefig(filename)
    plt.clf()
    plt.cla()
    plt.close()
    return [valid_num, total_num]

def main(args=None):
  args = parse_model_args(args)

  K.set_floatx('float64')

  run = get_run_id(**vars(args))
  log_dir = os.path.join(os.getcwd(), 'keras_logs', args.sim_type, run)
  weights_path = os.path.join(log_dir, 'weights.h5')

  if not os.path.exists(weights_path):
    raise FileNotFoundError(weights_path)

  X_fold, X_raw_fold, model, means, scales, wrong_units, args = survey_autoencoder(vars(args))

  print ("log_dir", log_dir)
  print("Weight matrix read...")

  full = joblib.load(args.survey_files[0])

  # Combine subclasses
  # Resulting in five classes: RR, EC, SR, M, ROT
  for lc in full:
    if ((lc.label == 'EW') or (lc.label == 'EA') or (lc.label == 'EB')):
      lc.label = 'ECL'
    if ((lc.label == 'CWA') or (lc.label == 'CWB') or (lc.label == 'DCEP') or
        (lc.label == 'DCEPS') or (lc.label == 'RVA')):
      lc.label = "CEPH"
    if ((lc.label == 'DSCT') or (lc.label == 'HADS')):
      lc.label = "DSCT"
    if ((lc.label == 'RRD') or (lc.label == 'RRC')):
      lc.label = "RRCD"
  top_classes = ['SR', 'RRAB', 'RRCD', 'M', 'ROT', 'ECL', 'CEPH', 'DSCT']
  new_classes = ['VAR']

  top = [lc for lc in full if lc.label in top_classes]
  new  = [lc for lc in full if lc.label in new_classes]

  # Preparation for classification probability
  classprob_pkl = joblib.load("./data/asassn/class_probs.pkl")
  class_probability = dict(classprob_pkl)

  if args.ss_resid:
    top = [lc for lc in top if lc.ss_resid <= args.ss_resid]

  if args.class_prob:
    top = [lc for lc in top if float(class_probability[lc.name.split("/")[-1][2:-4]]) >= 0.9]
    #top = [lc for lc in top if lc.class_prob >= args.class_prob]

  split = [el for lc in top for el in lc.split(args.n_min, args.n_max)]
  split_new = [el for lc in new for el in lc.split(args.n_min, args.n_max)]

  if args.period_fold:
    for lc in split:
      lc.period_fold()
    for lc in split_new:
      if lc.p is not None:
        lc.period_fold()
        
  X_list = [np.c_[lc.times, lc.measurements, lc.errors] for lc in split]
  classnames, indices = np.unique([lc.label for lc in split], return_inverse=True)
  y = classnames[indices]
  periods = np.array([np.log10(lc.p) for lc in split])
  periods = periods.reshape(len(split), 1)

  X_raw = pad_sequences(X_list, value=0., dtype='float64', padding='post')
  X, means, scales, wrong_units, X_err = preprocess(X_raw, args.m_max)
  y = y[~wrong_units]
  periods = periods[~wrong_units]

  train, valid = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED).split(X, y))[0]

  X_train = X[train]
  y_train = y[train]
  means_train = means[train]
  scales_train = scales[train]
  periods_train = periods[train]
  energy_dummy = np.zeros((X_train.shape[0], 1))

  X_valid = X[valid]
  y_valid = y[valid]
  means_valid = means[valid]
  scales_valid = scales[valid]
  periods_valid = periods[valid]
  energy_dummy_valid = np.zeros((X_valid.shape[0], 1))

  supports_train = np.concatenate((means_train, scales_train, periods_train), axis=1)
  supports_valid = np.concatenate((means_valid, scales_valid, periods_valid), axis=1)

  # New class data (VAR type)
  X_list_new = [np.c_[lc.times, lc.measurements, lc.errors] for lc in split_new]
  classnames_new, indices_new = np.unique([lc.label for lc in split_new], return_inverse=True)
  y_new = classnames_new[indices_new]
  periods_new = np.array([np.log10(lc.p) if lc.p is not None else 99.0 for lc in split_new])
  periods_new = periods_new.reshape(len(split_new), 1)

  X_raw_new = pad_sequences(X_list_new, value=0., dtype='float64', padding='post')
  X_new, means_new, scales_new, wrong_units_new, X_err_new = preprocess(X_raw_new, args.m_max)
  y_new = y_new[~wrong_units_new]
  periods_new = periods_new[~wrong_units_new]
  supports_new = np.concatenate((means_new, scales_new, periods_new), axis=1)

  ### Concatenating validation data and data from new classes for testing novelty detection
  y_new = np.concatenate((y_valid, y_new), axis=0)
  X_new = np.concatenate((X_valid, X_new), axis=0)
  supports_new = np.concatenate((supports_valid, supports_new), axis=0)

  num_supports_train = supports_train.shape[-1]
  num_supports_valid = supports_valid.shape[-1]
  assert(num_supports_train == num_supports_valid)
  num_additional = num_supports_train + 2 # 2 for reconstruction error

  ### flagging novel samples as 1, samples in superclasses 0.
  true_flags = np.array([1 if (l not in top_classes) else 0 for l in y_new])

  ### Making one-hot labels
  label_encoder1 = LabelEncoder()
  label_encoder1.fit(y_train)
  train_y_encoded = label_encoder1.transform(y_train)
  train_y   = np_utils.to_categorical(train_y_encoded)

  label_encoder2 = LabelEncoder()
  label_encoder2.fit(y_valid)
  valid_y_encoded = label_encoder2.transform(y_valid)
  valid_y   = np_utils.to_categorical(valid_y_encoded)

  label_encoder3 = LabelEncoder()
  label_encoder3.fit(y_new)
  new_y_encoded = label_encoder3.transform(y_new)
  new_y   = np_utils.to_categorical(new_y_encoded)


  ### Loading trained autoencoder network
  encode_model = Model(inputs=model.input, outputs=model.get_layer('encoding').output)
  decode_model = Model(inputs=model.input, outputs=model.get_layer('time_dist').output)

  X_train = np.float64(X_train)
  X_valid = np.float64(X_valid)
  X_new = np.float64(X_new)

  # Passing samples through trained layers
  encoding_train = encode_model.predict({'main_input': X_train, 'aux_input': np.delete(X_train, 1, axis=2), 'support_input':supports_train})
  encoding_valid = encode_model.predict({'main_input': X_valid, 'aux_input': np.delete(X_valid, 1, axis=2), 'support_input':supports_valid})
  encoding_new   = encode_model.predict({'main_input': X_new,   'aux_input': np.delete(X_new, 1, axis=2),   'support_input':supports_new})

  decoding_train = decode_model.predict({'main_input': X_train, 'aux_input': np.delete(X_train, 1, axis=2), 'support_input':supports_train})
  decoding_valid = decode_model.predict({'main_input': X_valid, 'aux_input': np.delete(X_valid, 1, axis=2), 'support_input':supports_valid})
  decoding_new   = decode_model.predict({'main_input': X_new, 'aux_input': np.delete(X_new, 1, axis=2),   'support_input':supports_new})

  z_both_train = extract_features([X_train[:,:,1], decoding_train[:,:,0], encoding_train, supports_train])
  z_both_valid = extract_features([X_valid[:,:,1], decoding_valid[:,:,0], encoding_valid, supports_valid])
  z_both_new   = extract_features([X_new[:,:,1],   decoding_new[:,:,0],   encoding_new, supports_new])

  z_both_train = K.eval(z_both_train)
  z_both_valid = K.eval(z_both_valid)
  z_both_new   = K.eval(z_both_new)

  # Retrieve estimation network if gmm is on
  if (args.gmm_on):
    estnet_model  = Model(inputs=model.input, outputs=model.get_layer('gamma').output)
    gamma_train = estnet_model.predict({'main_input': X_train, 'aux_input': np.delete(X_train, 1, axis=2), 'support_input':supports_train})
    gamma_valid = estnet_model.predict({'main_input': X_valid, 'aux_input': np.delete(X_valid, 1, axis=2), 'support_input':supports_valid})
  else:
    est_net = EstimationNet(args.estnet_size, args.num_classes, K.tanh)

  # Fit data to gmm if joint, train gmm if sequential
  gmm = GMM(args.num_classes, args.embedding+num_additional)
  gmm_init = gmm.init_gmm_variables()

  # If sequential training, create and train the estimation net
  if (not args.gmm_on):
    z_input = Input(shape=(args.embedding+num_additional,), name='z_input')
    gamma = est_net.inference(z_input, args.estnet_drop_frac)

    sigma_i = Lambda(lambda x: gmm.fit(x[0], x[1]),
                     output_shape=(args.num_classes, args.embedding+num_additional, args.embedding+num_additional),
                     name="sigma_i")([z_input, gamma])

    energy = Lambda(lambda x: gmm.energy(x[0], x[1]), name="energy")([z_input, sigma_i])

    # Setting up the GMM model
    model_output = [gamma, energy]
    model = Model(z_input, model_output)

    optimizer = Adam(lr=args.lr if not args.finetune_rate else args.finetune_rate)
    model.compile(optimizer=optimizer,
                  loss=['categorical_crossentropy', energy_loss],
                  metrics={'gamma':'accuracy'},
                  loss_weights=[1.0, args.lambda1])

    # Controlling outputs
    estnet_size = args.estnet_size
    estsize = "_".join(str(s) for s in estnet_size)
    gmm_run = 'estnet{}_estdrop{}_l1{}'.format(estsize, int(100*args.estnet_drop_frac), args.lambda1)
    gmm_dir = os.path.join(os.getcwd(), 'keras_logs', args.sim_type, run, gmm_run)
    if (not os.path.isdir(gmm_dir)):
      os.makedirs(gmm_dir)

    # For classifier training only
    args.nb_epoch = args.cls_epoch

    history = ku.train_and_log(z_both_train, {'gamma':train_y, 'energy':energy_dummy},
                               gmm_dir, model,
                               validation_data=(z_both_valid,
                                               {'gamma':valid_y, 'energy':energy_dummy_valid},
                                               {'gamma':None, 'energy':None}), **vars(args))

    gamma_train, energy_train = model.predict(z_both_train)
    gamma_valid, energy_valid = model.predict(z_both_valid)

  if (not args.gmm_on):
    log_dir = gmm_dir

  plot_dir = os.path.join(log_dir, 'figures/')
  if (not os.path.isdir(plot_dir)):
    os.makedirs(plot_dir)

  # Converting to Keras variables to use Keras.backend functions
  z_both_train_K = K.variable(z_both_train)
  gamma_train_K = K.variable(gamma_train)

  z_both_valid_K = K.variable(z_both_valid)
  gamma_valid_K = K.variable(gamma_valid)

  z_both_new_K = K.variable(z_both_new)

  # Fitting GMM parameters only with training set
  sigma_i_train = gmm.fit(z_both_train_K, gamma_train_K)
  sigma_i_dummy = 1.0
  assert (gmm.fitted == True)

  # Energy calculation
  energy_train = K.eval(gmm.energy(z_both_train_K, sigma_i_train))[:, 0]
  energy_valid = K.eval(gmm.energy(z_both_valid_K, sigma_i_dummy))[:, 0]
  energy_new   = K.eval(gmm.energy(z_both_new_K, sigma_i_dummy))[:, 0]

  energy_known = [e for i, e in enumerate(energy_new) if (true_flags[i] == 0)]
  energy_unknown = [e for i, e in enumerate(energy_new) if (true_flags[i] == 1)]
  print ("known/unknown", len(energy_known), len(energy_unknown))

  gmm_phi = K.eval(gmm.phi)
  gmm_mu = K.eval(gmm.mu)
  gmm_sigma = K.eval(gmm.sigma)

  np.savez(log_dir+'/gmm_parameters.npz', gmm_phi=gmm_phi, gmm_mu=gmm_mu, gmm_sigma=gmm_sigma)

  percentile_list = [80.0, 95.0]

  txtfilename = log_dir + "/novel_detection_scores.txt"
  txtfile = open(txtfilename, 'w')

  for per in percentile_list:
    new_class_energy_threshold = np.percentile(energy_train, per)
    print (f"Energy threshold to detect new class: {new_class_energy_threshold:.2f}")

    new_pred_flag = np.where(energy_new >= new_class_energy_threshold, 1, 0)

    prec, recall, fscore, _ = precision_recall_fscore_support(true_flags, new_pred_flag, average="binary")
  
    print (f"Detecting new using {per:.1f}% percentile")
    print(f" Precision = {prec:.3f}")
    print(f" Recall    = {recall:.3f}")
    print(f" F1-Score  = {fscore:.3f}")

    txtfile.write(f"Detecting new using {per:.1f}% percentile \n")
    txtfile.write(f" Precision = {prec:.3f}\n")
    txtfile.write(f" Recall    = {recall:.3f}\n")
    txtfile.write(f" F1-Score  = {fscore:.3f}\n")
  txtfile.close() 
 

  ### Make plots of energy
  nbin = 100
  fig = plt.figure()
  ax = fig.add_subplot(111)
  plt.hist(energy_train, nbin, normed=True, color='black',
           histtype='step', label='Training Set')
  plt.hist(np.isfinite(energy_unknown), nbin, normed=True, color='blue',
           histtype='step', label='Unknown Classes')
  plt.hist(energy_known, nbin, normed=True, color='green',
           histtype='step', label='Known Classes')
  plt.legend()
  plt.xlabel(r"Energy E(z)")
  plt.ylabel("Probability")
  plt.savefig(plot_dir+'energy_histogram.pdf', dpi=300,
              bbox_inches='tight')
  plt.clf()
  plt.cla()
  plt.close()


  ### Generate confusion matrix
  le_list = list(label_encoder2.classes_)
  predicted_onehot = gamma_valid 
  predicted_labels = [le_list[np.argmax(onehot, axis=None, out=None)] for onehot in predicted_onehot]

  corr_num, tot_num = plot_confusion_matrix(y_valid, predicted_labels, classnames,
                                            plot_dir+'asassn_nn_confusion.pdf')
  nn_acc = corr_num / tot_num

  ### Generate confusion matrix for RF
  RF_PARAM_GRID = {'n_estimators': [50, 100, 250], 'criterion': ['gini', 'entropy'],
                   'max_features': [0.05, 0.1, 0.2, 0.3], 'min_samples_leaf': [1, 2, 3]}
  rf_model = GridSearchCV(RandomForestClassifier(random_state=0), RF_PARAM_GRID)
  rf_model.fit(encoding_train, y[train])

  rf_train_acc = 100 * rf_model.score(encoding_train, y[train])
  rf_valid_acc = 100 * rf_model.score(encoding_valid, y[valid])

  plot_confusion_matrix(y[valid], rf_model.predict(encoding_valid), classnames,
                        plot_dir+'asassn_rf_confusion.pdf')

  ### Text output
  txtfilename = log_dir + "/classification_accuracy.txt"
  txtfile = open(txtfilename, 'w')

  ### Writing results
  txtfile.write("===== Classification Accuracy =====\n")
  txtfile.write(f"Neural Network Classifier: {nn_acc:.2f}\n")
  txtfile.write("==========================\n")
  txtfile.write("Random Forest Classifier\n")
  txtfile.write(f"Training accuracy: {rf_train_acc:2.2f}%\n")
  txtfile.write(f"Validation accuracy: {rf_valid_acc:2.2f}%\n")
  txtfile.write(f"Best RF {rf_model.best_params_}\n")
  txtfile.write("==========================\n")
  txtfile.close()


if __name__ == '__main__':
    args = main()
