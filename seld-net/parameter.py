# Parameters used in the feature extraction, neural network model, and training the SELDnet can be changed here.
#
# Ideally, do not change the values of the default parameters. Create separate cases with unique <task-id> as seen in
# the code below (if-else loop) and use them. This way you can easily reproduce a configuration on a later time.


def get_params(argv):
    print("SET: {}".format(argv))
    # ########### default parameters ##############
    params = dict(
        quick_test=False,    # To do quick test. Trains/test on small subset of dataset
        azi_only=True,      # Estimate Azimuth only

        # Dataset loading parameters
        dataset='ansim',    # Dataset to use: ansim, resim, cansim, cresim, real, mansim or mreal
        overlap=1,         # maximum number of overlapping sound events [1, 2, 3]
        split=1,           # Cross validation split [1, 2, 3]
        db=30,             # SNR of sound events.
        nfft=512,          # FFT/window length size

        # DNN Model parameters
        sequence_length=512,        # Feature sequence length
        batch_size=16,              # Batch size
        dropout_rate=0.0,           # Dropout rate, constant for all layers
        nb_cnn2d_filt=64,           # Number of CNN nodes, constant for each layer
        pool_size=[8, 8, 2],        # CNN pooling, length of list = number of CNN layers, list value = pooling per layer
        rnn_size=[128, 128],        # RNN contents, length of list = number of layers, list value = number of nodes
        fnn_size=[128],             # FNN contents, length of list = number of layers, list value = number of nodes
        loss_weights=[1., 50.],     # [sed, doa] weight for scaling the DNN outputs
        xyz_def_zero=True,          # Use default DOA Cartesian value x,y,z = 0,0,0
        nb_epochs=1000,             # Train for maximum epochs

        # Not important
        mode='regr',        # Only regression ('regr') supported as of now
        nb_cnn3d_filt=32,   # For future. Not relevant for now
        cnn_3d=False,       # For future. Not relevant for now
        weakness=0          # For future. Not relevant for now
    )
    params['patience'] = int(0.1 * params['nb_epochs'])     # Stop training if patience reached

    # ########### User defined parameters ##############
    if argv == '1':
        print("USING DEFAULT PARAMETERS\n")

    # Quick test
    elif argv == '999':
        print("QUICK TEST MODE\n")
        params['quick_test'] = True
        params['nb_epochs'] = 2

    # Different datasets
    elif argv == '2':  # anechoic simulated Ambisonic data set
        params['dataset'] = 'ansim'
        params['batch_size'] = 2
        params['sequence_length'] = 512

    elif argv == '3':  # reverberant simulated Ambisonic data set
        params['dataset'] = 'resim'
        params['sequence_length'] = 256

    elif argv == '4':  # anechoic simulated circular-array data set
        params['dataset'] = 'cansim'
        params['sequence_length'] = 256

    elif argv == '5':  # reverberant simulated circular-array data set
        params['dataset'] = 'cresim'
        params['sequence_length'] = 256

    elif argv == '6':  # real-life Ambisonic data set
        params['dataset'] = 'real'
        params['sequence_length'] = 512

    # anechoic circular array data set split 1, overlap 3
    elif argv == '7':  #
        params['dataset'] = 'cansim'
        params['overlap'] = 3
        params['split'] = 1

    # anechoic Ambisonic data set with sequence length 64 and batch size 32
    elif argv == '8':  #
        params['dataset'] = 'ansim'
        params['sequence_length'] = 64
        params['batch_size'] = 32

    elif argv == '9':
        params['dataset'] = 'drone_filtered'
        params['batch_size'] = 2
        params['sequence_length'] = 512
        params['quick_test'] = False
        params['azi_only'] = False
        params['db'] = 50
        # params['dropout_rate'] = 0.01 # to simulate audio dropoff caused by misaligned inmp441

    elif argv == '10':
        params['dataset'] = 'drone'
        params['batch_size'] = 4
        params['sequence_length'] = 512
        params['quick_test'] = False
        params['azi_only'] = False
        params['db'] = 50
    
    elif argv == '11':
        params['dataset'] = '1m_8channel_1class_drone'
        params['batch_size'] = 2
        params['sequence_length'] = 512
        params['quick_test'] = False
        params['azi_only'] = False
        params['db'] = 50

    elif argv == '12':
        params['dataset'] = '1m_8channel_1class'
        params['batch_size'] = 2
        params['sequence_length'] = 512
        params['quick_test'] = False
        params['azi_only'] = False
        params['db'] = 50

    elif argv == '13':
        params['dataset'] = '0.002m_8channel_1class'
        params['batch_size'] = 2
        params['sequence_length'] = 512
        params['quick_test'] = False
        params['azi_only'] = True
        params['db'] = 50
        params['loss_weights'] = [1., 400.]

    elif argv == '14':
        params['dataset'] = '0.120m_8channel_3class'
        params['batch_size'] = 1
        params['sequence_length'] = 512
        params['quick_test'] = False
        params['azi_only'] = True
        params['db'] = 50
        params['nb_cnn2d_filt'] = 128
        params['loss_weights'] = [1., 100.]
        params['pool_size'] = [8, 8, 4]
        params['rnn_size'] = [256, 256]
        params['fnn_size'] = [256]

    elif argv == '15':  # anechoic simulated Ambisonic data set
        params['dataset'] = 'ansim'
        params['batch_size'] = 2
        params['sequence_length'] = 512
        params['quick_test'] = False
        params['azi_only'] = True
        params['db'] = 30

    elif argv == '16':  # ANSIM clone
        params['dataset'] = 'ansim_clone'
        params['batch_size'] = 2
        params['sequence_length'] = 512
        params['loss_weights'] = [1., 50.]
        params['quick_test'] = False
        params['azi_only'] = True
        params['db'] = 30

    elif argv == '17':
        params['dataset'] = 'ansim_clone_0.120m'
        params['batch_size'] = 2
        params['sequence_length'] = 512
        params['loss_weights'] = [1., 50.]
        params['quick_test'] = False
        params['azi_only'] = True
        params['db'] = 30

    elif argv == '18':
        params['dataset'] = 'ansim_clone_0.120m_whistle'
        params['batch_size'] = 2
        params['sequence_length'] = 512
        params['loss_weights'] = [1., 50.]
        params['quick_test'] = False
        params['azi_only'] = True
        params['db'] = 30

    elif argv == '19':
        params['dataset'] = 'ansim_clone_0.120m_whistle_drone'
        params['batch_size'] = 2
        params['sequence_length'] = 512
        params['loss_weights'] = [1., 50.]
        params['quick_test'] = False
        params['azi_only'] = True
        params['db'] = 30

    elif argv == '20':
        params['dataset'] = 'criset_0.120m_3class_motor'
        params['batch_size'] = 2
        params['sequence_length'] = 512
        params['loss_weights'] = [1., 50.]
        params['quick_test'] = False
        params['azi_only'] = True
        params['db'] = 30

    else:
        print('ERROR: unknown argument {}'.format(argv))
        exit()

    for key, value in params.items():
        print("{}: {}".format(key, value))
    return params
