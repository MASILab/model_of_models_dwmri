from keras.backend.tensorflow_backend import set_session
import tensorflow as tf
from keras import backend as K

from keras.layers import Input, Dense, Dropout, concatenate, Conv3D, Flatten, Lambda
from keras.models import Model
from keras.layers import Activation, Add
from keras.wrappers.scikit_learn import KerasRegressor

from keras.optimizers import SGD, nadam, Adagrad, RMSprop, Adam
from keras.regularizers import l1,l2
from keras.callbacks import EarlyStopping, CSVLogger

#from utils.metrics import calc_acc, frac_loss, sh_loss

def build_sh_patch_resnet_bottleneck(bottleneck_n):

    # Patch Size is hard code in the network
    input_dims = (3, 3, 3, 45)
    inputs = Input(shape=input_dims)

    # First Convolution
    x1 = Conv3D(filters=45, kernel_size=10, strides=(1, 1, 1), padding='same')(inputs)

    # Functional Blocks
    sh0 = Lambda(lambda x: x[:, :, :, :, 0:1])(x1)
    sh2 = Lambda(lambda x: x[:, :, :, :, 1:6])(x1)
    sh4 = Lambda(lambda x: x[:, :, :, :, 6:15])(x1)
    sh6 = Lambda(lambda x: x[:, :, :, :, 16:28])(x1)
    sh8 = Lambda(lambda x: x[:, :, :, :, 28:45])(x1)

    sh0_c1 = Conv3D(filters=1, kernel_size=10, strides=(1, 1, 1), padding='same', activation='relu')(sh0)
    sh2_c1 = Conv3D(filters=5, kernel_size=10, strides=(1, 1, 1), padding='same', activation='relu')(sh2)
    sh4_c1 = Conv3D(filters=9, kernel_size=10, strides=(1, 1, 1), padding='same', activation='relu')(sh4)
    sh6_c1 = Conv3D(filters=13, kernel_size=10, strides=(1, 1, 1), padding='same', activation='relu')(sh6)
    sh8_c1 = Conv3D(filters=17, kernel_size=10, strides=(1, 1, 1), padding='same', activation='relu')(sh8)

    sh0_c2 = Conv3D(filters=1, kernel_size=10, strides=(1, 1, 1), padding='same', activation='relu')(sh0_c1)
    sh2_c2 = Conv3D(filters=5, kernel_size=10, strides=(1, 1, 1), padding='same', activation='relu')(sh2_c1)
    sh4_c2 = Conv3D(filters=9, kernel_size=10, strides=(1, 1, 1), padding='same', activation='relu')(sh4_c1)
    sh6_c2 = Conv3D(filters=13, kernel_size=10, strides=(1, 1, 1), padding='same', activation='relu')(sh6_c1)
    sh8_c2 = Conv3D(filters=17, kernel_size=10, strides=(1, 1, 1), padding='same', activation='relu')(sh8_c1)

    sh0_c3 = Conv3D(filters=1, kernel_size=10, strides=(1, 1, 1), padding='same')(sh0_c2)
    sh2_c3 = Conv3D(filters=5, kernel_size=10, strides=(1, 1, 1), padding='same')(sh2_c2)
    sh4_c3 = Conv3D(filters=9, kernel_size=10, strides=(1, 1, 1), padding='same')(sh4_c2)
    sh6_c3 = Conv3D(filters=13, kernel_size=10, strides=(1, 1, 1), padding='same')(sh6_c2)
    sh8_c3 = Conv3D(filters=17, kernel_size=10, strides=(1, 1, 1), padding='same')(sh8_c2)

    combined = concatenate([sh0_c3, sh2_c3, sh4_c3, sh6_c3, sh8_c3])
    x2 = Conv3D(filters=45, kernel_size=10, strides=(1, 1, 1), padding='same')(combined)

    # Complete Residual Block
    res_add = Add()([x1, x2])

    x3 = Conv3D(filters=45, kernel_size=10, strides=(1, 1, 1), padding='same', activation='relu')(res_add)
    x4 = Conv3D(filters=45, kernel_size=10, strides=(1, 1, 1), padding='same')(x3)

    x5 = Flatten()(x4)

    # x6 is the BOTTLENECK Layer, we pull out different decoders from the bottleneck
    x6 = Dense(bottleneck_n)(x5)

    ## TODO Different Decoder heads for different microstructure methods
    # Ball Stick 2003 -> Metric Count 5
    head_1_l1 = Dense(50, activation='relu')(x6)
    head_1_l2 = Dense(5, activation='linear', name='ball_stick')(head_1_l1)

    # IVIM -> Metric Count 3
    head_2_l1 = Dense(50, activation='relu')(x6)
    head_2_l2 = Dense(3, activation='linear', name='ivim')(head_2_l1)

    # MC SMT -> Metric Count 2
    head_3_l1 = Dense(50, activation='relu')(x6)
    head_3_l2 = Dense(2, activation='linear', name='mc_smt')(head_3_l1)

    # NODDI WATSON -> Metric Count 5
    head_4_l1 = Dense(50, activation='relu')(x6)
    head_4_l2 = Dense(5, activation='linear', name='noddi')(head_4_l1)

    # Extract Fractional Volume Output
    # f_out = Dense(3, activation='linear')(x5)
    # total_out = concatenate([x6, f_out])

    # Model define inputs and outputs from network structure
    model = Model(inputs=inputs, outputs=[head_1_l2, head_2_l2, head_3_l2, head_4_l2])

    opt_func = RMSprop(lr=0.0001)

    model.compile(loss={'ball_stick': 'mse',
                        'ivim': 'mse',
                        'mc_smt': 'mse',
                        'noddi': 'mse'},
                  loss_weights={'ball_stick': 1.0,
                                'ivim': 1.0,
                                'mc_smt': 1.0,
                                'noddi': 1.0},
                  optimizer=opt_func,
                  metrics={'ball_stick': 'mse',
                           'ivim': 'mse',
                           'mc_smt': 'mse',
                           'noddi': 'mse'})

    print(model.summary())
    return model

def build_patchnet_compart_orient(bottleneck_n):

    # Patch Size is hard code in the network
    input_dims = (3, 3, 3, 45)
    inputs = Input(shape=input_dims)

    # First Convolution
    x1 = Conv3D(filters=45, kernel_size=10, strides=(1, 1, 1), padding='same')(inputs)

    # Functional Blocks
    sh0 = Lambda(lambda x: x[:, :, :, :, 0:1])(x1)
    sh2 = Lambda(lambda x: x[:, :, :, :, 1:6])(x1)
    sh4 = Lambda(lambda x: x[:, :, :, :, 6:15])(x1)
    sh6 = Lambda(lambda x: x[:, :, :, :, 16:28])(x1)
    sh8 = Lambda(lambda x: x[:, :, :, :, 28:45])(x1)

    sh0_c1 = Conv3D(filters=1, kernel_size=10, strides=(1, 1, 1), padding='same', activation='relu')(sh0)
    sh2_c1 = Conv3D(filters=5, kernel_size=10, strides=(1, 1, 1), padding='same', activation='relu')(sh2)
    sh4_c1 = Conv3D(filters=9, kernel_size=10, strides=(1, 1, 1), padding='same', activation='relu')(sh4)
    sh6_c1 = Conv3D(filters=13, kernel_size=10, strides=(1, 1, 1), padding='same', activation='relu')(sh6)
    sh8_c1 = Conv3D(filters=17, kernel_size=10, strides=(1, 1, 1), padding='same', activation='relu')(sh8)

    sh0_c2 = Conv3D(filters=1, kernel_size=10, strides=(1, 1, 1), padding='same', activation='relu')(sh0_c1)
    sh2_c2 = Conv3D(filters=5, kernel_size=10, strides=(1, 1, 1), padding='same', activation='relu')(sh2_c1)
    sh4_c2 = Conv3D(filters=9, kernel_size=10, strides=(1, 1, 1), padding='same', activation='relu')(sh4_c1)
    sh6_c2 = Conv3D(filters=13, kernel_size=10, strides=(1, 1, 1), padding='same', activation='relu')(sh6_c1)
    sh8_c2 = Conv3D(filters=17, kernel_size=10, strides=(1, 1, 1), padding='same', activation='relu')(sh8_c1)

    sh0_c3 = Conv3D(filters=1, kernel_size=10, strides=(1, 1, 1), padding='same')(sh0_c2)
    sh2_c3 = Conv3D(filters=5, kernel_size=10, strides=(1, 1, 1), padding='same')(sh2_c2)
    sh4_c3 = Conv3D(filters=9, kernel_size=10, strides=(1, 1, 1), padding='same')(sh4_c2)
    sh6_c3 = Conv3D(filters=13, kernel_size=10, strides=(1, 1, 1), padding='same')(sh6_c2)
    sh8_c3 = Conv3D(filters=17, kernel_size=10, strides=(1, 1, 1), padding='same')(sh8_c2)

    combined = concatenate([sh0_c3, sh2_c3, sh4_c3, sh6_c3, sh8_c3])
    x2 = Conv3D(filters=45, kernel_size=10, strides=(1, 1, 1), padding='same')(combined)

    # Complete Residual Block
    res_add = Add()([x1, x2])

    x3 = Conv3D(filters=45, kernel_size=10, strides=(1, 1, 1), padding='same', activation='relu')(res_add)
    x4 = Conv3D(filters=45, kernel_size=10, strides=(1, 1, 1), padding='same')(x3)

    x5 = Flatten()(x4)

    # x6 is the BOTTLENECK Layer, we pull out different decoders from the bottleneck
    x6 = Dense(bottleneck_n)(x5)

    ## TODO Different Decoder heads for different microstructure methods
    # Ball Stick 2003 -> Metric Count 6
    head_1_l1 = Dense(50, activation='relu')(x6)
    head_1_l2 = Dense(6, activation='linear', name='ball_stick')(head_1_l1)

    # IVIM -> Metric Count 3
    head_2_l1 = Dense(50, activation='relu')(x6)
    head_2_l2 = Dense(3, activation='linear', name='ivim')(head_2_l1)

    # MC SMT -> Metric Count 2
    head_3_l1 = Dense(50, activation='relu')(x6)
    head_3_l2 = Dense(2, activation='linear', name='mc_smt')(head_3_l1)

    # NODDI WATSON -> Metric Count 6
    head_4_l1 = Dense(50, activation='relu')(x6)
    head_4_l2 = Dense(6, activation='linear', name='noddi')(head_4_l1)

    # DTI -> Metric Count 2
    head_5_l1 = Dense(50, activation='relu')(x6)
    head_5_l2 = Dense(2, activation='linear', name='dti')(head_5_l1)

    # MT-CSD -> Metric Count 45
    head_6_l1 = Dense(50, activation='relu')(x6)
    head_6_l2 = Dense(45, activation='linear', name='mt_csd')(head_6_l1)

    # Extract Fractional Volume Output
    # f_out = Dense(3, activation='linear')(x5)
    # total_out = concatenate([x6, f_out])

    # Model define inputs and outputs from network structure
    model = Model(inputs=inputs, outputs=[head_1_l2, head_2_l2, head_3_l2, head_4_l2, head_5_l2, head_6_l2])

    opt_func = RMSprop(lr=0.0001)

    model.compile(loss={'ball_stick': 'mse',
                        'ivim': 'mse',
                        'mc_smt': 'mse',
                        'noddi': 'mse',
                        'dti': 'mse',
                        'mt_csd': 'mse'},
                  loss_weights={'ball_stick': 1.0,
                                'ivim': 1.0,
                                'mc_smt': 1.0,
                                'noddi': 1.0,
                                'dti': 1.0,
                                'mt_csd': 1.0
                                },
                  optimizer=opt_func,
                  metrics={'ball_stick': 'mse',
                           'ivim': 'mse',
                           'mc_smt': 'mse',
                           'noddi': 'mse',
                           'dti': 'mse',
                           'mt_csd': 'mse'
                           })

    print(model.summary())
    return model

def build_patchnet_compart_dti(bottleneck_n):

    # Patch Size is hard code in the network
    input_dims = (3, 3, 3, 45)
    inputs = Input(shape=input_dims)

    # First Convolution
    x1 = Conv3D(filters=45, kernel_size=10, strides=(1, 1, 1), padding='same')(inputs)

    # Functional Blocks
    sh0 = Lambda(lambda x: x[:, :, :, :, 0:1])(x1)
    sh2 = Lambda(lambda x: x[:, :, :, :, 1:6])(x1)
    sh4 = Lambda(lambda x: x[:, :, :, :, 6:15])(x1)
    sh6 = Lambda(lambda x: x[:, :, :, :, 16:28])(x1)
    sh8 = Lambda(lambda x: x[:, :, :, :, 28:45])(x1)

    sh0_c1 = Conv3D(filters=1, kernel_size=10, strides=(1, 1, 1), padding='same', activation='relu')(sh0)
    sh2_c1 = Conv3D(filters=5, kernel_size=10, strides=(1, 1, 1), padding='same', activation='relu')(sh2)
    sh4_c1 = Conv3D(filters=9, kernel_size=10, strides=(1, 1, 1), padding='same', activation='relu')(sh4)
    sh6_c1 = Conv3D(filters=13, kernel_size=10, strides=(1, 1, 1), padding='same', activation='relu')(sh6)
    sh8_c1 = Conv3D(filters=17, kernel_size=10, strides=(1, 1, 1), padding='same', activation='relu')(sh8)

    sh0_c2 = Conv3D(filters=1, kernel_size=10, strides=(1, 1, 1), padding='same', activation='relu')(sh0_c1)
    sh2_c2 = Conv3D(filters=5, kernel_size=10, strides=(1, 1, 1), padding='same', activation='relu')(sh2_c1)
    sh4_c2 = Conv3D(filters=9, kernel_size=10, strides=(1, 1, 1), padding='same', activation='relu')(sh4_c1)
    sh6_c2 = Conv3D(filters=13, kernel_size=10, strides=(1, 1, 1), padding='same', activation='relu')(sh6_c1)
    sh8_c2 = Conv3D(filters=17, kernel_size=10, strides=(1, 1, 1), padding='same', activation='relu')(sh8_c1)

    sh0_c3 = Conv3D(filters=1, kernel_size=10, strides=(1, 1, 1), padding='same')(sh0_c2)
    sh2_c3 = Conv3D(filters=5, kernel_size=10, strides=(1, 1, 1), padding='same')(sh2_c2)
    sh4_c3 = Conv3D(filters=9, kernel_size=10, strides=(1, 1, 1), padding='same')(sh4_c2)
    sh6_c3 = Conv3D(filters=13, kernel_size=10, strides=(1, 1, 1), padding='same')(sh6_c2)
    sh8_c3 = Conv3D(filters=17, kernel_size=10, strides=(1, 1, 1), padding='same')(sh8_c2)

    combined = concatenate([sh0_c3, sh2_c3, sh4_c3, sh6_c3, sh8_c3])
    x2 = Conv3D(filters=45, kernel_size=10, strides=(1, 1, 1), padding='same')(combined)

    # Complete Residual Block
    res_add = Add()([x1, x2])

    x3 = Conv3D(filters=45, kernel_size=10, strides=(1, 1, 1), padding='same', activation='relu')(res_add)
    x4 = Conv3D(filters=45, kernel_size=10, strides=(1, 1, 1), padding='same')(x3)

    x5 = Flatten()(x4)

    # x6 is the BOTTLENECK Layer, we pull out different decoders from the bottleneck
    x6 = Dense(bottleneck_n)(x5)

    ## TODO Different Decoder heads for different microstructure methods
    # Ball Stick 2003 -> Metric Count 6
    head_1_l1 = Dense(50, activation='relu')(x6)
    head_1_l2 = Dense(6, activation='linear', name='ball_stick')(head_1_l1)

    # IVIM -> Metric Count 3
    head_2_l1 = Dense(50, activation='relu')(x6)
    head_2_l2 = Dense(3, activation='linear', name='ivim')(head_2_l1)

    # MC SMT -> Metric Count 2
    head_3_l1 = Dense(50, activation='relu')(x6)
    head_3_l2 = Dense(2, activation='linear', name='mc_smt')(head_3_l1)

    # NODDI WATSON -> Metric Count 6
    head_4_l1 = Dense(50, activation='relu')(x6)
    head_4_l2 = Dense(6, activation='linear', name='noddi')(head_4_l1)

    # DTI -> Metric Count 2
    head_5_l1 = Dense(50, activation='relu')(x6)
    head_5_l2 = Dense(2, activation='linear', name='dti')(head_5_l1)

    # Extract Fractional Volume Output
    # f_out = Dense(3, activation='linear')(x5)
    # total_out = concatenate([x6, f_out])

    # Model define inputs and outputs from network structure
    model = Model(inputs=inputs, outputs=[head_1_l2, head_2_l2, head_3_l2, head_4_l2, head_5_l2])

    opt_func = RMSprop(lr=0.0001)

    model.compile(loss={'ball_stick': 'mse',
                        'ivim': 'mse',
                        'mc_smt': 'mse',
                        'noddi': 'mse',
                        'dti': 'mse'
                        },
                  loss_weights={'ball_stick': 1.0,
                                'ivim': 1.0,
                                'mc_smt': 1.0,
                                'noddi': 1.0,
                                'dti': 1.0
                                },
                  optimizer=opt_func,
                  metrics={'ball_stick': 'mse',
                           'ivim': 'mse',
                           'mc_smt': 'mse',
                           'noddi': 'mse',
                           'dti': 'mse'
                           })

    print(model.summary())
    return model

def build_patch_double_bottleneck(bottleneck_n):

    # Patch Size is hard code in the network
    input_dims = (3, 3, 3, 45)
    inputs = Input(shape=input_dims)

    # First Convolution
    x1 = Conv3D(filters=45, kernel_size=10, strides=(1, 1, 1), padding='same')(inputs)

    # Functional Blocks
    sh0 = Lambda(lambda x: x[:, :, :, :, 0:1])(x1)
    sh2 = Lambda(lambda x: x[:, :, :, :, 1:6])(x1)
    sh4 = Lambda(lambda x: x[:, :, :, :, 6:15])(x1)
    sh6 = Lambda(lambda x: x[:, :, :, :, 16:28])(x1)
    sh8 = Lambda(lambda x: x[:, :, :, :, 28:45])(x1)

    sh0_c1 = Conv3D(filters=1, kernel_size=10, strides=(1, 1, 1), padding='same', activation='relu')(sh0)
    sh2_c1 = Conv3D(filters=5, kernel_size=10, strides=(1, 1, 1), padding='same', activation='relu')(sh2)
    sh4_c1 = Conv3D(filters=9, kernel_size=10, strides=(1, 1, 1), padding='same', activation='relu')(sh4)
    sh6_c1 = Conv3D(filters=13, kernel_size=10, strides=(1, 1, 1), padding='same', activation='relu')(sh6)
    sh8_c1 = Conv3D(filters=17, kernel_size=10, strides=(1, 1, 1), padding='same', activation='relu')(sh8)

    sh0_c2 = Conv3D(filters=1, kernel_size=10, strides=(1, 1, 1), padding='same', activation='relu')(sh0_c1)
    sh2_c2 = Conv3D(filters=5, kernel_size=10, strides=(1, 1, 1), padding='same', activation='relu')(sh2_c1)
    sh4_c2 = Conv3D(filters=9, kernel_size=10, strides=(1, 1, 1), padding='same', activation='relu')(sh4_c1)
    sh6_c2 = Conv3D(filters=13, kernel_size=10, strides=(1, 1, 1), padding='same', activation='relu')(sh6_c1)
    sh8_c2 = Conv3D(filters=17, kernel_size=10, strides=(1, 1, 1), padding='same', activation='relu')(sh8_c1)

    sh0_c3 = Conv3D(filters=1, kernel_size=10, strides=(1, 1, 1), padding='same')(sh0_c2)
    sh2_c3 = Conv3D(filters=5, kernel_size=10, strides=(1, 1, 1), padding='same')(sh2_c2)
    sh4_c3 = Conv3D(filters=9, kernel_size=10, strides=(1, 1, 1), padding='same')(sh4_c2)
    sh6_c3 = Conv3D(filters=13, kernel_size=10, strides=(1, 1, 1), padding='same')(sh6_c2)
    sh8_c3 = Conv3D(filters=17, kernel_size=10, strides=(1, 1, 1), padding='same')(sh8_c2)

    combined = concatenate([sh0_c3, sh2_c3, sh4_c3, sh6_c3, sh8_c3])
    x2 = Conv3D(filters=45, kernel_size=10, strides=(1, 1, 1), padding='same')(combined)

    # Complete Residual Block
    res_add = Add()([x1, x2])

    x3 = Conv3D(filters=45, kernel_size=10, strides=(1, 1, 1), padding='same', activation='relu')(res_add)
    x4 = Conv3D(filters=45, kernel_size=10, strides=(1, 1, 1), padding='same')(x3)

    #TODO  Sam Bottleneck convolutional/multidimensional manifold
    #bottleneck_dims = (3, 3, 3)
    #s1 = Conv3D(filters=1, kernel_size=bottleneck_dims, strides=1, padding='same')(x4)
    #x6 = Flatten(s1)

    x5 = Flatten()(x4)
    #
    # # x6 is the BOTTLENECK Layer, we pull out different decoders from the bottleneck
    x5 = Dense(bottleneck_n)(x5)
    x6 = Dense(bottleneck_n)(x5)

    ## TODO Different Decoder heads for different microstructure methods
    # Ball Stick 2003 -> Metric Count 6
    head_1_l1 = Dense(50, activation='relu')(x6)
    head_1_l2 = Dense(6, activation='linear', name='ball_stick')(head_1_l1)

    # IVIM -> Metric Count 3
    head_2_l1 = Dense(50, activation='relu')(x6)
    head_2_l2 = Dense(3, activation='linear', name='ivim')(head_2_l1)

    # MC SMT -> Metric Count 2
    head_3_l1 = Dense(50, activation='relu')(x6)
    head_3_l2 = Dense(2, activation='linear', name='mc_smt')(head_3_l1)

    # NODDI WATSON -> Metric Count 6
    head_4_l1 = Dense(50, activation='relu')(x6)
    head_4_l2 = Dense(6, activation='linear', name='noddi')(head_4_l1)

    # DTI -> Metric Count 2
    head_5_l1 = Dense(50, activation='relu')(x6)
    head_5_l2 = Dense(2, activation='linear', name='dti')(head_5_l1)

    # Extract Fractional Volume Output
    # f_out = Dense(3, activation='linear')(x5)
    # total_out = concatenate([x6, f_out])

    # Model define inputs and outputs from network structure
    model = Model(inputs=inputs, outputs=[head_1_l2, head_2_l2, head_3_l2, head_4_l2, head_5_l2])

    opt_func = RMSprop(lr=0.0001)

    model.compile(loss={'ball_stick': 'mse',
                        'ivim': 'mse',
                        'mc_smt': 'mse',
                        'noddi': 'mse',
                        'dti': 'mse'
                        },
                  loss_weights={'ball_stick': 1.0,
                                'ivim': 1.0,
                                'mc_smt': 1.0,
                                'noddi': 1.0,
                                'dti': 1.0
                                },
                  optimizer=opt_func,
                  metrics={'ball_stick': 'mse',
                           'ivim': 'mse',
                           'mc_smt': 'mse',
                           'noddi': 'mse',
                           'dti': 'mse'
                           })

    print(model.summary())
    return model

def build_just_dti_double_bottleneck(bottleneck_n):
    # Patch Size is hard code in the network
    input_dims = (3, 3, 3, 45)
    inputs = Input(shape=input_dims)

    # First Convolution
    x1 = Conv3D(filters=45, kernel_size=3, strides=(1, 1, 1), padding='same')(inputs)

    # Functional Blocks
    sh0 = Lambda(lambda x: x[:, :, :, :, 0:1])(x1)
    sh2 = Lambda(lambda x: x[:, :, :, :, 1:6])(x1)
    sh4 = Lambda(lambda x: x[:, :, :, :, 6:15])(x1)
    sh6 = Lambda(lambda x: x[:, :, :, :, 16:28])(x1)
    sh8 = Lambda(lambda x: x[:, :, :, :, 28:45])(x1)

    sh0_c1 = Conv3D(filters=1, kernel_size=3, strides=(1, 1, 1), padding='same', activation='elu')(sh0)
    sh2_c1 = Conv3D(filters=5, kernel_size=3, strides=(1, 1, 1), padding='same', activation='elu')(sh2)
    sh4_c1 = Conv3D(filters=9, kernel_size=3, strides=(1, 1, 1), padding='same', activation='elu')(sh4)
    sh6_c1 = Conv3D(filters=13, kernel_size=3, strides=(1, 1, 1), padding='same', activation='elu')(sh6)
    sh8_c1 = Conv3D(filters=17, kernel_size=3, strides=(1, 1, 1), padding='same', activation='elu')(sh8)

    sh0_c2 = Conv3D(filters=1, kernel_size=3, strides=(1, 1, 1), padding='same', activation='elu')(sh0_c1)
    sh2_c2 = Conv3D(filters=5, kernel_size=3, strides=(1, 1, 1), padding='same', activation='elu')(sh2_c1)
    sh4_c2 = Conv3D(filters=9, kernel_size=3, strides=(1, 1, 1), padding='same', activation='elu')(sh4_c1)
    sh6_c2 = Conv3D(filters=13, kernel_size=3, strides=(1, 1, 1), padding='same', activation='elu')(sh6_c1)
    sh8_c2 = Conv3D(filters=17, kernel_size=3, strides=(1, 1, 1), padding='same', activation='elu')(sh8_c1)

    sh0_c3 = Conv3D(filters=1, kernel_size=3, strides=(1, 1, 1), padding='same')(sh0_c2)
    sh2_c3 = Conv3D(filters=5, kernel_size=3, strides=(1, 1, 1), padding='same')(sh2_c2)
    sh4_c3 = Conv3D(filters=9, kernel_size=3, strides=(1, 1, 1), padding='same')(sh4_c2)
    sh6_c3 = Conv3D(filters=13, kernel_size=3, strides=(1, 1, 1), padding='same')(sh6_c2)
    sh8_c3 = Conv3D(filters=17, kernel_size=3, strides=(1, 1, 1), padding='same')(sh8_c2)

    combined = concatenate([sh0_c3, sh2_c3, sh4_c3, sh6_c3, sh8_c3])
    x2 = Conv3D(filters=45, kernel_size=3, strides=(1, 1, 1), padding='same')(combined)

    # Complete Residual Block
    res_add = Add()([x1, x2])

    x3 = Conv3D(filters=45, kernel_size=3, strides=(1, 1, 1), padding='same', activation='elu')(res_add)
    x4 = Conv3D(filters=45, kernel_size=3, strides=(1, 1, 1), padding='same')(x3)

    # TODO  Sam Bottleneck convolutional/multidimensional manifold
    # bottleneck_dims = (3, 3, 3)
    # s1 = Conv3D(filters=1, kernel_size=bottleneck_dims, strides=1, padding='same')(x4)
    # x6 = Flatten(s1)

    x5 = Flatten()(x4)
    #
    # # x6 is the BOTTLENECK Layer, we pull out different decoders from the bottleneck
    x5 = Dense(bottleneck_n)(x5)
    x6 = Dense(bottleneck_n)(x5)

    ## TODO Different Decoder heads for different microstructure methods
    # Ball Stick 2003 -> Metric Count 6
    #head_1_l1 = Dense(50, activation='elu')(x6)
    #head_1_l2 = Dense(6, activation='linear', name='ball_stick')(head_1_l1)

    # IVIM -> Metric Count 3
    #head_2_l1 = Dense(50, activation='relu')(x6)
    #head_2_l2 = Dense(3, activation='linear', name='ivim')(head_2_l1)

    # MC SMT -> Metric Count 2
    #head_3_l1 = Dense(50, activation='relu')(x6)
    #head_3_l2 = Dense(2, activation='linear', name='mc_smt')(head_3_l1)

    # NODDI WATSON -> Metric Count 6
    #head_4_l1 = Dense(50, activation='relu')(x6)
    #head_4_l2 = Dense(6, activation='linear', name='noddi')(head_4_l1)

    # DTI -> Metric Count 2
    head_5_l1 = Dense(50, activation='relu')(x6)
    head_5_l2 = Dense(2, activation='linear', name='dti')(head_5_l1)

    # Extract Fractional Volume Output
    # f_out = Dense(3, activation='linear')(x5)
    # total_out = concatenate([x6, f_out])

    # Model define inputs and outputs from network structure
    model = Model(inputs=inputs, outputs=head_5_l2)

    opt_func = RMSprop(lr=0.0001)

    model.compile(
                  loss={'dti': 'mse'},
                  loss_weights={'dti': 1.0},
                  optimizer=opt_func,
                  metrics={'dti': 'mse'}
                  )

    print(model.summary())
    return model

def build_just_ivim_double_bottleneck(bottleneck_n):
    # Patch Size is hard code in the network
    input_dims = (3, 3, 3, 45)
    inputs = Input(shape=input_dims)

    # First Convolution
    x1 = Conv3D(filters=45, kernel_size=3, strides=(1, 1, 1), padding='same')(inputs)

    # Functional Blocks
    sh0 = Lambda(lambda x: x[:, :, :, :, 0:1])(x1)
    sh2 = Lambda(lambda x: x[:, :, :, :, 1:6])(x1)
    sh4 = Lambda(lambda x: x[:, :, :, :, 6:15])(x1)
    sh6 = Lambda(lambda x: x[:, :, :, :, 16:28])(x1)
    sh8 = Lambda(lambda x: x[:, :, :, :, 28:45])(x1)

    sh0_c1 = Conv3D(filters=1, kernel_size=3, strides=(1, 1, 1), padding='same', activation='elu')(sh0)
    sh2_c1 = Conv3D(filters=5, kernel_size=3, strides=(1, 1, 1), padding='same', activation='elu')(sh2)
    sh4_c1 = Conv3D(filters=9, kernel_size=3, strides=(1, 1, 1), padding='same', activation='elu')(sh4)
    sh6_c1 = Conv3D(filters=13, kernel_size=3, strides=(1, 1, 1), padding='same', activation='elu')(sh6)
    sh8_c1 = Conv3D(filters=17, kernel_size=3, strides=(1, 1, 1), padding='same', activation='elu')(sh8)

    sh0_c2 = Conv3D(filters=1, kernel_size=3, strides=(1, 1, 1), padding='same', activation='elu')(sh0_c1)
    sh2_c2 = Conv3D(filters=5, kernel_size=3, strides=(1, 1, 1), padding='same', activation='elu')(sh2_c1)
    sh4_c2 = Conv3D(filters=9, kernel_size=3, strides=(1, 1, 1), padding='same', activation='elu')(sh4_c1)
    sh6_c2 = Conv3D(filters=13, kernel_size=3, strides=(1, 1, 1), padding='same', activation='elu')(sh6_c1)
    sh8_c2 = Conv3D(filters=17, kernel_size=3, strides=(1, 1, 1), padding='same', activation='elu')(sh8_c1)

    sh0_c3 = Conv3D(filters=1, kernel_size=3, strides=(1, 1, 1), padding='same')(sh0_c2)
    sh2_c3 = Conv3D(filters=5, kernel_size=3, strides=(1, 1, 1), padding='same')(sh2_c2)
    sh4_c3 = Conv3D(filters=9, kernel_size=3, strides=(1, 1, 1), padding='same')(sh4_c2)
    sh6_c3 = Conv3D(filters=13, kernel_size=3, strides=(1, 1, 1), padding='same')(sh6_c2)
    sh8_c3 = Conv3D(filters=17, kernel_size=3, strides=(1, 1, 1), padding='same')(sh8_c2)

    combined = concatenate([sh0_c3, sh2_c3, sh4_c3, sh6_c3, sh8_c3])
    x2 = Conv3D(filters=45, kernel_size=3, strides=(1, 1, 1), padding='same')(combined)

    # Complete Residual Block
    res_add = Add()([x1, x2])

    x3 = Conv3D(filters=45, kernel_size=3, strides=(1, 1, 1), padding='same', activation='elu')(res_add)
    x4 = Conv3D(filters=45, kernel_size=3, strides=(1, 1, 1), padding='same')(x3)

    # TODO  Sam Bottleneck convolutional/multidimensional manifold
    # bottleneck_dims = (3, 3, 3)
    # s1 = Conv3D(filters=1, kernel_size=bottleneck_dims, strides=1, padding='same')(x4)
    # x6 = Flatten(s1)

    x5 = Flatten()(x4)
    #
    # # x6 is the BOTTLENECK Layer, we pull out different decoders from the bottleneck
    x5 = Dense(bottleneck_n)(x5)
    x6 = Dense(bottleneck_n)(x5)

    ## TODO Different Decoder heads for different microstructure methods
    # Ball Stick 2003 -> Metric Count 6
    #head_1_l1 = Dense(50, activation='elu')(x6)
    #head_1_l2 = Dense(6, activation='linear', name='ball_stick')(head_1_l1)

    # IVIM -> Metric Count 3
    head_2_l1 = Dense(50, activation='relu')(x6)
    head_2_l2 = Dense(3, activation='linear', name='ivim')(head_2_l1)

    # MC SMT -> Metric Count 2
    #head_3_l1 = Dense(50, activation='relu')(x6)
    #head_3_l2 = Dense(2, activation='linear', name='mc_smt')(head_3_l1)

    # NODDI WATSON -> Metric Count 6
    #head_4_l1 = Dense(50, activation='relu')(x6)
    #head_4_l2 = Dense(6, activation='linear', name='noddi')(head_4_l1)

    # DTI -> Metric Count 2
    #head_5_l1 = Dense(50, activation='relu')(x6)
    #head_5_l2 = Dense(2, activation='linear', name='dti')(head_5_l1)

    # Extract Fractional Volume Output
    # f_out = Dense(3, activation='linear')(x5)
    # total_out = concatenate([x6, f_out])

    # Model define inputs and outputs from network structure
    model = Model(inputs=inputs, outputs=head_2_l2)

    opt_func = RMSprop(lr=0.0001)

    model.compile(
                  loss={'ivim': 'mse'},
                  loss_weights={'ivim': 1.0},
                  optimizer=opt_func,
                  metrics={'ivim': 'mse'}
                  )

    print(model.summary())
    return model

def build_just_mcsmt_double_bottleneck(bottleneck_n):
    # Patch Size is hard code in the network
    input_dims = (3, 3, 3, 45)
    inputs = Input(shape=input_dims)

    # First Convolution
    x1 = Conv3D(filters=45, kernel_size=3, strides=(1, 1, 1), padding='same')(inputs)

    # Functional Blocks
    sh0 = Lambda(lambda x: x[:, :, :, :, 0:1])(x1)
    sh2 = Lambda(lambda x: x[:, :, :, :, 1:6])(x1)
    sh4 = Lambda(lambda x: x[:, :, :, :, 6:15])(x1)
    sh6 = Lambda(lambda x: x[:, :, :, :, 16:28])(x1)
    sh8 = Lambda(lambda x: x[:, :, :, :, 28:45])(x1)

    sh0_c1 = Conv3D(filters=1, kernel_size=3, strides=(1, 1, 1), padding='same', activation='elu')(sh0)
    sh2_c1 = Conv3D(filters=5, kernel_size=3, strides=(1, 1, 1), padding='same', activation='elu')(sh2)
    sh4_c1 = Conv3D(filters=9, kernel_size=3, strides=(1, 1, 1), padding='same', activation='elu')(sh4)
    sh6_c1 = Conv3D(filters=13, kernel_size=3, strides=(1, 1, 1), padding='same', activation='elu')(sh6)
    sh8_c1 = Conv3D(filters=17, kernel_size=3, strides=(1, 1, 1), padding='same', activation='elu')(sh8)

    sh0_c2 = Conv3D(filters=1, kernel_size=3, strides=(1, 1, 1), padding='same', activation='elu')(sh0_c1)
    sh2_c2 = Conv3D(filters=5, kernel_size=3, strides=(1, 1, 1), padding='same', activation='elu')(sh2_c1)
    sh4_c2 = Conv3D(filters=9, kernel_size=3, strides=(1, 1, 1), padding='same', activation='elu')(sh4_c1)
    sh6_c2 = Conv3D(filters=13, kernel_size=3, strides=(1, 1, 1), padding='same', activation='elu')(sh6_c1)
    sh8_c2 = Conv3D(filters=17, kernel_size=3, strides=(1, 1, 1), padding='same', activation='elu')(sh8_c1)

    sh0_c3 = Conv3D(filters=1, kernel_size=3, strides=(1, 1, 1), padding='same')(sh0_c2)
    sh2_c3 = Conv3D(filters=5, kernel_size=3, strides=(1, 1, 1), padding='same')(sh2_c2)
    sh4_c3 = Conv3D(filters=9, kernel_size=3, strides=(1, 1, 1), padding='same')(sh4_c2)
    sh6_c3 = Conv3D(filters=13, kernel_size=3, strides=(1, 1, 1), padding='same')(sh6_c2)
    sh8_c3 = Conv3D(filters=17, kernel_size=3, strides=(1, 1, 1), padding='same')(sh8_c2)

    combined = concatenate([sh0_c3, sh2_c3, sh4_c3, sh6_c3, sh8_c3])
    x2 = Conv3D(filters=45, kernel_size=3, strides=(1, 1, 1), padding='same')(combined)

    # Complete Residual Block
    res_add = Add()([x1, x2])

    x3 = Conv3D(filters=45, kernel_size=3, strides=(1, 1, 1), padding='same', activation='elu')(res_add)
    x4 = Conv3D(filters=45, kernel_size=3, strides=(1, 1, 1), padding='same')(x3)

    # TODO  Sam Bottleneck convolutional/multidimensional manifold
    # bottleneck_dims = (3, 3, 3)
    # s1 = Conv3D(filters=1, kernel_size=bottleneck_dims, strides=1, padding='same')(x4)
    # x6 = Flatten(s1)

    x5 = Flatten()(x4)
    #
    # # x6 is the BOTTLENECK Layer, we pull out different decoders from the bottleneck
    x5 = Dense(bottleneck_n)(x5)
    x6 = Dense(bottleneck_n)(x5)

    ## TODO Different Decoder heads for different microstructure methods
    # Ball Stick 2003 -> Metric Count 6
    #head_1_l1 = Dense(50, activation='elu')(x6)
    #head_1_l2 = Dense(6, activation='linear', name='ball_stick')(head_1_l1)

    # IVIM -> Metric Count 3
    #head_2_l1 = Dense(50, activation='relu')(x6)
    #head_2_l2 = Dense(3, activation='linear', name='ivim')(head_2_l1)

    # MC SMT -> Metric Count 2
    head_3_l1 = Dense(50, activation='relu')(x6)
    head_3_l2 = Dense(1, activation='relu', name='mc_smt_1')(head_3_l1)
    head_3_l3 = Dense(1, activation='relu', name='mc_smt_2')(head_3_l1)

    # NODDI WATSON -> Metric Count 6
    #head_4_l1 = Dense(50, activation='relu')(x6)
    #head_4_l2 = Dense(6, activation='linear', name='noddi')(head_4_l1)

    # DTI -> Metric Count 2
    #head_5_l1 = Dense(50, activation='relu')(x6)
    #head_5_l2 = Dense(2, activation='linear', name='dti')(head_5_l1)

    # Extract Fractional Volume Output
    # f_out = Dense(3, activation='linear')(x5)
    # total_out = concatenate([x6, f_out])

    # Model define inputs and outputs from network structure
    model = Model(inputs=inputs, outputs=[head_3_l2, head_3_l3])

    opt_func = RMSprop(lr=0.0001)

    model.compile(
                  loss={'mc_smt_1': 'mae',
                        'mc_smt_2': 'mae'},
                  loss_weights={'mc_smt_1': 1.0,
                                'mc_smt_2': 1.0},
                  optimizer=opt_func,
                  metrics={'mc_smt_1': 'mae',
                           'mc_smt_2': 'mae'}
                  )

    print(model.summary())
    return model


def build_dti_ivim_double_bottleneck(bottleneck_n):
    # Patch Size is hard code in the network
    input_dims = (3, 3, 3, 45)
    inputs = Input(shape=input_dims)

    # First Convolution
    x1 = Conv3D(filters=45, kernel_size=3, strides=(1, 1, 1), padding='same')(inputs)

    # Functional Blocks
    sh0 = Lambda(lambda x: x[:, :, :, :, 0:1])(x1)
    sh2 = Lambda(lambda x: x[:, :, :, :, 1:6])(x1)
    sh4 = Lambda(lambda x: x[:, :, :, :, 6:15])(x1)
    sh6 = Lambda(lambda x: x[:, :, :, :, 16:28])(x1)
    sh8 = Lambda(lambda x: x[:, :, :, :, 28:45])(x1)

    sh0_c1 = Conv3D(filters=1, kernel_size=3, strides=(1, 1, 1), padding='same', activation='elu')(sh0)
    sh2_c1 = Conv3D(filters=5, kernel_size=3, strides=(1, 1, 1), padding='same', activation='elu')(sh2)
    sh4_c1 = Conv3D(filters=9, kernel_size=3, strides=(1, 1, 1), padding='same', activation='elu')(sh4)
    sh6_c1 = Conv3D(filters=13, kernel_size=3, strides=(1, 1, 1), padding='same', activation='elu')(sh6)
    sh8_c1 = Conv3D(filters=17, kernel_size=3, strides=(1, 1, 1), padding='same', activation='elu')(sh8)

    sh0_c2 = Conv3D(filters=1, kernel_size=3, strides=(1, 1, 1), padding='same', activation='elu')(sh0_c1)
    sh2_c2 = Conv3D(filters=5, kernel_size=3, strides=(1, 1, 1), padding='same', activation='elu')(sh2_c1)
    sh4_c2 = Conv3D(filters=9, kernel_size=3, strides=(1, 1, 1), padding='same', activation='elu')(sh4_c1)
    sh6_c2 = Conv3D(filters=13, kernel_size=3, strides=(1, 1, 1), padding='same', activation='elu')(sh6_c1)
    sh8_c2 = Conv3D(filters=17, kernel_size=3, strides=(1, 1, 1), padding='same', activation='elu')(sh8_c1)

    sh0_c3 = Conv3D(filters=1, kernel_size=3, strides=(1, 1, 1), padding='same')(sh0_c2)
    sh2_c3 = Conv3D(filters=5, kernel_size=3, strides=(1, 1, 1), padding='same')(sh2_c2)
    sh4_c3 = Conv3D(filters=9, kernel_size=3, strides=(1, 1, 1), padding='same')(sh4_c2)
    sh6_c3 = Conv3D(filters=13, kernel_size=3, strides=(1, 1, 1), padding='same')(sh6_c2)
    sh8_c3 = Conv3D(filters=17, kernel_size=3, strides=(1, 1, 1), padding='same')(sh8_c2)

    combined = concatenate([sh0_c3, sh2_c3, sh4_c3, sh6_c3, sh8_c3])
    x2 = Conv3D(filters=45, kernel_size=3, strides=(1, 1, 1), padding='same')(combined)

    # Complete Residual Block
    res_add = Add()([x1, x2])

    x3 = Conv3D(filters=45, kernel_size=3, strides=(1, 1, 1), padding='same', activation='elu')(res_add)
    x4 = Conv3D(filters=45, kernel_size=3, strides=(1, 1, 1), padding='same')(x3)

    # TODO  Sam Bottleneck convolutional/multidimensional manifold
    # bottleneck_dims = (3, 3, 3)
    # s1 = Conv3D(filters=1, kernel_size=bottleneck_dims, strides=1, padding='same')(x4)
    # x6 = Flatten(s1)

    x5 = Flatten()(x4)
    #
    # # x6 is the BOTTLENECK Layer, we pull out different decoders from the bottleneck
    x5 = Dense(bottleneck_n)(x5)
    x6 = Dense(bottleneck_n)(x5)

    ## TODO Different Decoder heads for different microstructure methods
    # Ball Stick 2003 -> Metric Count 6
    #head_1_l1 = Dense(50, activation='elu')(x6)
    #head_1_l2 = Dense(6, activation='linear', name='ball_stick')(head_1_l1)

    # IVIM -> Metric Count 3
    head_2_l1 = Dense(50, activation='relu')(x6)
    head_2_l2 = Dense(3, activation='linear', name='ivim')(head_2_l1)

    # MC SMT -> Metric Count 2
    #head_3_l1 = Dense(50, activation='relu')(x6)
    #head_3_l2 = Dense(2, activation='linear', name='mc_smt')(head_3_l1)

    # NODDI WATSON -> Metric Count 6
    #head_4_l1 = Dense(50, activation='relu')(x6)
    #head_4_l2 = Dense(6, activation='linear', name='noddi')(head_4_l1)

    # DTI -> Metric Count 2
    head_5_l1 = Dense(50, activation='relu')(x6)
    head_5_l2 = Dense(2, activation='linear', name='dti')(head_5_l1)

    # Extract Fractional Volume Output
    # f_out = Dense(3, activation='linear')(x5)
    # total_out = concatenate([x6, f_out])

    # Model define inputs and outputs from network structure
    model = Model(inputs=inputs, outputs=[head_2_l2, head_5_l2])

    opt_func = RMSprop(lr=0.0001)

    model.compile(loss={'ivim': 'mse',
                        'dti': 'mse'
                        },
                  loss_weights={'ivim': 1.0,
                                'dti': 1.0
                                },
                  optimizer=opt_func,
                  metrics={'ivim': 'mse',
                           'dti': 'mse'
                           })

    print(model.summary())
    return model
