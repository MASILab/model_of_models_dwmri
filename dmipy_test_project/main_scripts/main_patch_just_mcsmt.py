import os
import numpy as np
import random
import argparse
import time
import json
import pickle
import tensorflow as tf

from keras.callbacks import TensorBoard, ModelCheckpoint

from models.bottleneck_patch_models import build_just_mcsmt_double_bottleneck
from data_generators.data_generator_microstructure_patch import nifti_just_mcsmt_generator
from data_generators.test_data_generator_microstructure_patch import test_just_mcsmt_predictor

# Set Seed First Priority,
seed_value = 1203
np.random.seed(seed_value)
random.seed(seed_value)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_list', '-d', required=False, type=str,
                        default=r'/nfs/masi/nathv/miccai_2020_hcp_100_jsonfiles/data_list_hcp_100_ivim_dti.json',
                        help='Data List file for training stored in a JSON format')

    parser.add_argument('--model_dir', '-m', required=False, type=str,
                        default=r'/nfs/masi/nathv/miccai_2020_hcp_100_experiments/double_bottleneck_just_mcsmt_sep_mae',
                        help='model output directory')

    parser.add_argument('--save_dir', '-s', required=False, type=str,
                        default=r'/nfs/masi/nathv/miccai_2020_hcp_100_experiments/double_bottleneck_just_mcsmt_sep_mae_test',
                        help='Saving Directory for the testing data')

    parser.add_argument('--script_mode', required=False, type=str,
                        default=r'train_test',
                        help='Yet to be defined')

    parser.add_argument('--prior_weights', required=False, type=str,
                        default=r'',
                        help='Yet to be defined')

    args = parser.parse_args()

    # Print out the arguments passed in
    for arg in vars(args):
        print('Argument Detected {}'.format(arg))
        print(getattr(args, arg))

    # Create the Model directory if non-existent
    model_base_path = args.model_dir
    model_base_path = os.path.normpath(model_base_path)
    if os.path.exists(model_base_path) == False:
        os.mkdir(model_base_path)

    # Create the saving directory if non-existent
    save_base_path = args.save_dir
    save_base_path = os.path.normpath(save_base_path)
    if os.path.exists(save_base_path) == False:
        os.mkdir(save_base_path)

    #
    bottleneck_values = [50]

    ## TODO We loop here for different values of the bottleneck
    for each_bnval in bottleneck_values:

        print('Training Bottleneck model with {} neurons'.format(each_bnval))

        bn_dir_name = 'bn_' + str(each_bnval)
        bn_model_path = os.path.join(model_base_path, bn_dir_name)
        if os.path.exists(bn_model_path) == False:
            os.mkdir(bn_model_path)

        bn_save_path = os.path.join(save_base_path, bn_dir_name)
        if os.path.exists(bn_save_path) == False:
            os.mkdir(bn_save_path)

        # Load Json file
        data_list_path = args.data_list
        data_list_path = os.path.normpath(data_list_path)
        all_data = json.load(open(data_list_path))
        tr_data = all_data["train"]
        val_data = all_data["validation"]

        # TODO Just Test on 3 Volumes for Now
        test_data = all_data["test"]
        test_data = test_data[0:3]

        bottleneck_model = build_just_mcsmt_double_bottleneck(each_bnval)

        patch_crop = [3, 3, 3]

        model_ckpt_name = "weights-improvement.hdf5"
        model_ckpt_path = os.path.join(bn_model_path, model_ckpt_name)
        checkpoint = ModelCheckpoint(model_ckpt_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')


        # Data Generators
        trainGen = nifti_just_mcsmt_generator(tr_data, bs=1000, patch_size=patch_crop)
        validGen = nifti_just_mcsmt_generator(val_data, bs=1000, patch_size=patch_crop)

        # Callbacks for Tensorboard, Saving model with checkpointing
        tensor_board = TensorBoard(log_dir=bn_model_path)

        # model_ckpt_name = "weights-improvement-{epoch:02d}-{val_loss:.2f}.hdf5"

        callbacks_list = [checkpoint, tensor_board]

        # Fit the DataGenerators
        history = bottleneck_model.fit_generator(generator=trainGen,
                                                 validation_data=validGen,
                                                 steps_per_epoch=500,
                                                 validation_steps=300,
                                                 epochs=30,
                                                 verbose=1,
                                                 callbacks=callbacks_list)

        metric_dump = history.history
        f_pkl_path = os.path.join(bn_model_path, 'bn_{}_metric.pkl'.format(each_bnval))
        pickle.dump(metric_dump, open(f_pkl_path, 'wb'))

        print('Training Finished ...')

        print('Reloading Best Saved Weights')
        bottleneck_model.load_weights(model_ckpt_path)

        print('Testing Model on the unseen data')
        test_just_mcsmt_predictor(bottleneck_model, test_data, bn_save_path)

    return None


if __name__ == "__main__":
    main()
