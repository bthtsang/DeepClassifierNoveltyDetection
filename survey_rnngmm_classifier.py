import numpy as np
import joblib
from keras.layers import Input, LSTM, GRU, SimpleRNN
from keras.models import Model
from keras.preprocessing.sequence import pad_sequences

import keras_util as ku
from autoencoder import encoder, decoder, extract_features
from light_curve import LightCurve

# Added for GMM
from gmm import GMM
from estimation_net import EstimationNet
from keras.layers import Lambda

# Added for classifier training
import keras.backend as K
from sklearn.model_selection import StratifiedKFold
from keras.optimizers import Adam
# Added for one-hot conversion
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils

SEED = 0

def preprocess(X_raw, m_max=np.inf):
    X = X_raw.copy()

    wrong_units =  np.all(np.isnan(X[:, :, 1])) | (np.nanmax(X[:, :, 1], axis=1) > m_max)
    X = X[~wrong_units, :, :]

    # Replace times w/ lags
    X[:, :, 0] = ku.times_to_lags(X[:, :, 0])

    means = np.atleast_2d(np.nanmean(X[:, :, 1], axis=1)).T
    X[:, :, 1] -= means

    scales = np.atleast_2d(np.nanstd(X[:, :, 1], axis=1)).T
    X[:, :, 1] /= scales
    X[:, :, 2] /= scales # scale errors also

    X_err = X[:, :, 2] 
    # Drop_errors from input; only used as weights
    X = X[:, :, :2]

    return X, means, scales, wrong_units, X_err


def energy_loss(y_true, y_pred):
  return K.mean(y_pred)

def main(args=None):
    """Train an autoencoder model from `LightCurve` objects saved in
    `args.survey_files`.
    
    args: dict
        Dictionary of values to override default values in `keras_util.parse_model_args`;
        can also be passed via command line. See `parse_model_args` for full list of
        possible arguments.
    """
    args = ku.parse_model_args(args)

    np.random.seed(0)

    K.set_floatx('float64')

    run = ku.get_run_id(**vars(args))

    if not args.survey_files:
        raise ValueError("No survey files given")

    lc_lists = [joblib.load(f) for f in args.survey_files]
    n_reps = [max(len(y) for y in lc_lists) // len(x) for x in lc_lists]
    combined = sum([x * i for x, i in zip(lc_lists, n_reps)], [])

    # Preparation for classification probability
    classprob_pkl = joblib.load("./data/asassn/class_probs.pkl")
    class_probability = dict(classprob_pkl)

    # Combine subclasses into eight superclasses:
    # CEPH, DSCT, ECL, RRAB, RRCD, M, ROT, SR
    for lc in combined:
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

    print ("Number of raw LCs:", len(combined))

    if args.lomb_score:
        combined = [lc for lc in combined if lc.best_score >= args.lomb_score]
    if args.ss_resid:
        combined = [lc for lc in combined if lc.ss_resid <= args.ss_resid]
    if args.class_prob:
        combined = [lc for lc in combined if float(class_probability[lc.name.split("/")[-1][2:-4]]) >= args.class_prob]
        #combined = [lc for lc in combined if lc.class_prob >= args.class_prob]

    # Select only superclasses for training
    combined = [lc for lc in combined if lc.label in top_classes]

    split = [el for lc in combined for el in lc.split(args.n_min, args.n_max)]
    if args.period_fold:
        for lc in split:
            lc.period_fold()

    X_list = [np.c_[lc.times, lc.measurements, lc.errors] for lc in split]
    classnames, indices = np.unique([lc.label for lc in split], return_inverse=True)
    y = classnames[indices]
    periods = np.array([np.log10(lc.p) for lc in split])
    periods = periods.reshape(len(split), 1)

    X_raw = pad_sequences(X_list, value=np.nan, dtype='float64', padding='post')

    model_type_dict = {'gru': GRU, 'lstm': LSTM, 'vanilla': SimpleRNN}
    X, means, scales, wrong_units, X_err = preprocess(X_raw, args.m_max)

    y = y[~wrong_units]
    periods = periods[~wrong_units]

    # Prepare the indices for training and validation
    train, valid = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED).split(X, y))[0]

    X_valid = X[valid]
    y_valid = y[valid]
    means_valid = means[valid]
    scales_valid = scales[valid]
    periods_valid = periods[valid]
    energy_dummy_valid = np.zeros((X_valid.shape[0], 1))
    X_err_valid = X_err[valid]
    sample_weight_valid = 1. / X_err_valid

    X = X[train]
    y = y[train]
    means = means[train]
    scales = scales[train]
    periods = periods[train]
    energy_dummy = np.zeros((X.shape[0], 1))
    X_err = X_err[train]
    sample_weight = 1. / X_err

    supports_valid = np.concatenate((means_valid, scales_valid, periods_valid), axis=1)
    supports = np.concatenate((means, scales, periods), axis=1)

    num_supports_train = supports.shape[-1]
    num_supports_valid = supports_valid.shape[-1]
    assert(num_supports_train == num_supports_valid)
    num_additional = num_supports_train + 2 # 2 for reconstruction error

    if (args.gmm_on):
      gmm = GMM(args.num_classes, args.embedding+num_additional)

    ### Covert labels into one-hot vectors for training
    label_encoder = LabelEncoder()
    label_encoder.fit(y)
    ### Transform the integers into one-hot vector for softmax
    train_y_encoded = label_encoder.transform(y)
    train_y   = np_utils.to_categorical(train_y_encoded)
    ### Repeat for validation dataset 
    label_encoder = LabelEncoder()
    label_encoder.fit(y_valid)
    ### Transform the integers into one-hot vector for softmax
    valid_y_encoded = label_encoder.transform(y_valid)
    valid_y   = np_utils.to_categorical(valid_y_encoded)

    main_input = Input(shape=(X.shape[1], 2), name='main_input') # dim: (200, 2) = dt, mag
    aux_input  = Input(shape=(X.shape[1], 1), name='aux_input') # dim: (200, 1) = dt's

    model_input = [main_input, aux_input]
    if (args.gmm_on):
      support_input = Input(shape=(num_supports_train,), name='support_input') 
      model_input = [main_input, aux_input, support_input]

    encode = encoder(main_input, layer=model_type_dict[args.model_type], 
                     output_size=args.embedding, **vars(args))
    decode = decoder(encode, num_layers=args.decode_layers if args.decode_layers
                                                           else args.num_layers,
                     layer=model_type_dict[args.decode_type if args.decode_type
                                           else args.model_type],
                     n_step=X.shape[1], aux_input=aux_input,
                     **{k: v for k, v in vars(args).items() if k != 'num_layers'})
    optimizer = Adam(lr=args.lr if not args.finetune_rate else args.finetune_rate)

    if (not args.gmm_on):
      model = Model(model_input, decode)

      model.compile(optimizer=optimizer, loss='mse',
                    sample_weight_mode='temporal')

    else: 
      est_net = EstimationNet(args.estnet_size, args.num_classes, K.tanh)

      # extract x_i and hat{x}_{i} for reconstruction error feature calculations
      main_input_slice = Lambda(lambda x: x[:,:,1], name="main_slice")(main_input)
      decode_slice = Lambda(lambda x: x[:,:,0], name="decode_slice")(decode)
      z_both = Lambda(extract_features, name="concat_zs")([main_input_slice, decode_slice, encode, support_input])

      gamma = est_net.inference(z_both, args.estnet_drop_frac)
      print ("z_both shape", z_both.shape)
      print ("gamma shape", gamma.shape)

      sigma_i = Lambda(lambda x: gmm.fit(x[0], x[1]), 
                       output_shape=(args.num_classes, args.embedding+num_additional, args.embedding+num_additional), 
                       name="sigma_i")([z_both, gamma])

      energy = Lambda(lambda x: gmm.energy(x[0], x[1]), name="energy")([z_both, sigma_i])

      model_output = [decode, gamma, energy]

      model = Model(model_input, model_output)

      optimizer = Adam(lr=args.lr if not args.finetune_rate else args.finetune_rate)

      model.compile(optimizer=optimizer, 
                    loss=['mse', 'categorical_crossentropy', energy_loss],
                    loss_weights=[1.0, 1.0, args.lambda1],
                    metrics={'gamma':'accuracy'},
                    sample_weight_mode=['temporal', None, None])

      # summary for checking dimensions
      print(model.summary())


    if (args.gmm_on): 
      history = ku.train_and_log( {'main_input':X, 'aux_input':np.delete(X, 1, axis=2), 'support_input':supports},
                                  {'time_dist':X[:, :, [1]], 'gamma':train_y, 'energy':energy_dummy}, run, model, 
                                  sample_weight={'time_dist':sample_weight, 'gamma':None, 'energy':None},
                                  validation_data=(
                                  {'main_input':X_valid, 'aux_input':np.delete(X_valid, 1, axis=2), 'support_input':supports_valid}, 
                                  {'time_dist':X_valid[:, :, [1]], 'gamma':valid_y, 'energy':energy_dummy_valid},
                                  {'time_dist':sample_weight_valid, 'gamma':None, 'energy':None}), 
                                  **vars(args))
    else: # Just train RNN autoencoder
      history = ku.train_and_log({'main_input':X, 'aux_input':np.delete(X, 1, axis=2)},
                                  X[:, :, [1]], run, model,
                                  sample_weight=sample_weight,
                                  validation_data=(
                                  {'main_input':X_valid, 'aux_input':np.delete(X_valid, 1, axis=2)},
                                  X_valid[:, :, [1]], sample_weight_valid), **vars(args))

    return X, X_raw, model, means, scales, wrong_units, args


if __name__ == '__main__':
    X, X_raw, model, means, scales, wrong_units, args = main()
