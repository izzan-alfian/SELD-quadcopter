#
# A wrapper script that trains the SELDnet. The training stops when the SELD error (check paper) stops improving.
#

import os
import sys
import numpy as np
import matplotlib.pyplot as plot
import cls_data_generator
import evaluation_metrics
import keras_model
import parameter
import utils
import time
import tensorflow as tf
from IPython import embed
plot.switch_backend('agg')


def collect_test_labels(_data_gen_test, _data_out, classification_mode, quick_test):
    # Collecting ground truth for test data
    nb_batch = 2 if quick_test else _data_gen_test.get_total_batches_in_data()

    batch_size = _data_out[0][0]
    gt_sed = np.zeros((nb_batch * batch_size, _data_out[0][1], _data_out[0][2]))
    gt_doa = np.zeros((nb_batch * batch_size, _data_out[0][1], _data_out[1][2]))

    print("nb_batch in test: {}".format(nb_batch))
    cnt = 0
    for tmp_feat, tmp_label in _data_gen_test.generate():
        gt_sed[cnt * batch_size:(cnt + 1) * batch_size, :, :] = tmp_label[0]
        gt_doa[cnt * batch_size:(cnt + 1) * batch_size, :, :] = tmp_label[1]
        cnt = cnt + 1
        if cnt == nb_batch:
            break
    return gt_sed.astype(int), gt_doa


def plot_functions(fig_name, _tr_loss, _val_loss, _sed_loss, _doa_loss, _epoch_metric_loss, epoch_idx):
    
    plot.figure()
    plot.suptitle(('epoch index: ' + str(epoch_idx)), fontsize=16)
    
    current_epoch = epoch_idx + 1

    plot.subplot(311)
    plot.plot(range(current_epoch), _tr_loss, label='train loss')
    plot.plot(range(current_epoch), _val_loss, label='val loss')
    plot.legend()
    plot.grid(True)

    plot.subplot(312)
    plot.plot(range(current_epoch), _epoch_metric_loss, label='metric')
    plot.plot(range(current_epoch), _sed_loss[:, 0], label='er')
    plot.plot(range(current_epoch), _sed_loss[:, 1], label='f1')
    plot.legend()
    plot.grid(True)

    plot.subplot(313)
    plot.plot(range(current_epoch), _doa_loss[:, 1], label='gt_thres')
    plot.plot(range(current_epoch), _doa_loss[:, 2], label='pred_thres')
    plot.legend()
    plot.grid(True)

    plot.savefig(fig_name)
    plot.close()


def main(argv):
    """
    Main wrapper for training sound event localization and detection network.
    
    :param argv: expects two optional inputs. 
        first input: job_id - (optional) all the output files will be uniquely represented with this. (default) 1
        second input: task_id - (optional) To chose the system configuration in parameters.py. 
                                (default) uses default parameters
    """
    if len(argv) != 3:
        print('\n\n')
        print('-------------------------------------------------------------------------------------------------------')
        print('The code expected two inputs')
        print('\t>> python seld.py <job-id> <task-id>')
        print('\t\t<job-id> is a unique identifier which is used for output filenames (models, training plots). '
              'You can use any number or string for this.')
        print('\t\t<task-id> is used to choose the user-defined parameter set from parameter.py')
        print('Using default inputs for now')
        print('-------------------------------------------------------------------------------------------------------')
        print('\n\n')

    task_id = '1' if len(argv) < 3 else argv[-1]
    params = parameter.get_params(task_id)

    job_id = 1 if len(argv) < 2 else argv[1]

    model_dir = 'models/'
    utils.create_folder(model_dir)
    unique_name = '{}_ov{}_split{}_{}{}_3d{}_{}'.format(
        params['dataset'], params['overlap'], params['split'], params['mode'], params['weakness'],
        int(params['cnn_3d']), job_id
    )
    unique_name = os.path.join(model_dir, unique_name)
    print("unique_name: {}\n".format(unique_name))

    data_gen_train = cls_data_generator.DataGenerator(
        dataset=params['dataset'], ov=params['overlap'], split=params['split'], db=params['db'], nfft=params['nfft'],
        batch_size=params['batch_size'], seq_len=params['sequence_length'], classifier_mode=params['mode'],
        weakness=params['weakness'], datagen_mode='train', cnn3d=params['cnn_3d'], xyz_def_zero=params['xyz_def_zero'],
        azi_only=params['azi_only']
    )

    data_gen_test = cls_data_generator.DataGenerator(
        dataset=params['dataset'], ov=params['overlap'], split=params['split'], db=params['db'], nfft=params['nfft'],
        batch_size=params['batch_size'], seq_len=params['sequence_length'], classifier_mode=params['mode'],
        weakness=params['weakness'], datagen_mode='test', cnn3d=params['cnn_3d'], xyz_def_zero=params['xyz_def_zero'],
        azi_only=params['azi_only'], shuffle=False
    )

    data_in, data_out = data_gen_train.get_data_sizes()
    print(
        'FEATURES:\n'
        '\tdata_in: {}\n'
        '\tdata_out: {}\n'.format(
            data_in, data_out
        )
    )

    gt = collect_test_labels(data_gen_test, data_out, params['mode'], params['quick_test'])
    sed_gt = evaluation_metrics.reshape_3Dto2D(gt[0])
    doa_gt = evaluation_metrics.reshape_3Dto2D(gt[1])

    print(
        'MODEL:\n'
        '\tdropout_rate: {}\n'
        '\tCNN: nb_cnn_filt: {}, pool_size{}\n'
        '\trnn_size: {}, fnn_size: {}\n'.format(
            params['dropout_rate'],
            params['nb_cnn3d_filt'] if params['cnn_3d'] else params['nb_cnn2d_filt'], params['pool_size'],
            params['rnn_size'], params['fnn_size']
        )
    )

    model = tf.keras.models.load_model(unique_name + '_model.keras')
    model.summary()
    
    nb_epoch = 2 if params['quick_test'] else params['nb_epochs']

    class CustomEvaluation(tf.keras.callbacks.Callback):
        def __init__(self, data_gen_test, params, unique_name, sed_gt, doa_gt):
            super().__init__()
            self.data_gen_test = data_gen_test
            self.seld_params = params
            self.unique_name = unique_name
            self.sed_gt = sed_gt
            self.doa_gt = doa_gt
            self.best_metric = 99999
            self.best_epoch = -1
            # Store metrics
            self.all_tr_loss = []
            self.all_val_loss = []
            self.all_sed_loss = []
            self.all_doa_loss = []
            self.all_epoch_metric_loss = []


        def on_epoch_end(self, epoch, logs=None):
            # logs already contains train_loss and val_loss
            self.all_tr_loss.append(logs.get('loss'))
            self.all_val_loss.append(logs.get('val_loss'))

            # Run prediction
            pred = self.model.predict(
                self.data_gen_test.generate(),
                steps=2 if self.seld_params['quick_test'] else self.data_gen_test.get_total_batches_in_data(),
                verbose=0
            )

            # --- Your entire evaluation logic from the loop goes here ---
            sed_pred = evaluation_metrics.reshape_3Dto2D(pred[0]) > 0.5
            doa_pred = evaluation_metrics.reshape_3Dto2D(pred[1])
            
            current_sed_loss = evaluation_metrics.compute_sed_scores(sed_pred, self.sed_gt, self.data_gen_test.nb_frames_1s())
            
            if self.seld_params['azi_only']:
                current_doa_loss, conf_mat = evaluation_metrics.compute_doa_scores_regr_xy(doa_pred, self.doa_gt, sed_pred, self.sed_gt)
            else:
                current_doa_loss, conf_mat = evaluation_metrics.compute_doa_scores_regr_xyz(doa_pred, self.doa_gt, sed_pred, self.sed_gt)
            
            current_metric = np.mean([
                current_sed_loss[0],
                1 - current_sed_loss[1],
                2 * np.arcsin(current_doa_loss[1] / 2.0) / np.pi,
                1 - (current_doa_loss[5] / float(self.doa_gt.shape[0]))]
            )
            
            self.all_sed_loss.append(current_sed_loss)
            self.all_doa_loss.append(current_doa_loss)
            self.all_epoch_metric_loss.append(current_metric)
            # --- End of evaluation logic ---

            print(f"Epoch {epoch+1}: custom_metric={current_metric:.4f}")

            # Check for improvement and save the best model
            if current_metric < self.best_metric:
                self.best_metric = current_metric
                self.best_epoch = epoch
                self.model.save(f'{self.unique_name}_model.keras')
                # Set patience attribute in the callback
                self.patience_cnt = 0 
            else:
                self.patience_cnt += 1

            # Early stopping logic
            if self.patience_cnt > self.seld_params['patience']:
                self.model.stop_training = True
    
    
    custom_eval_callback = CustomEvaluation(data_gen_test, params, unique_name, sed_gt, doa_gt)
    
    model.fit(
        data_gen_train.generate(),
        steps_per_epoch=2 if params['quick_test'] else data_gen_train.get_total_batches_in_data(),
        validation_data=data_gen_test.generate(),
        validation_steps=2 if params['quick_test'] else data_gen_test.get_total_batches_in_data(),
        epochs=nb_epoch,
        callbacks=[custom_eval_callback], # Add the callback here
        verbose=2 # Use verbose=2 for one-line-per-epoch logging
    )

    # After training, you can plot using the data stored in the callback
    plot_functions(
        unique_name + '.png',
        custom_eval_callback.all_tr_loss,
        custom_eval_callback.all_val_loss,
        np.array(custom_eval_callback.all_sed_loss),
        np.array(custom_eval_callback.all_doa_loss),
        custom_eval_callback.all_epoch_metric_loss
    )


if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv))
    except (ValueError, IOError) as e:
            sys.exit(e)
