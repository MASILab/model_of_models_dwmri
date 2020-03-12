import os
import numpy as np
import random
import argparse
import time
import json
import tensorflow as tf

from keras.callbacks import TensorBoard, ModelCheckpoint

from models.bottleneck_patch_models import build_sh_patch_resnet_bottleneck
from data_generators.data_generator_microstructure_patch import nifti_image_generator_patch

# Set Seed First Priority,
seed_value = 1203
np.random.seed(seed_value)
random.seed(seed_value)

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_list', '-d', required=False, type=str,
                        default=r'/nfs/masi/nathv/miccai_2020/micro_methods_hcp_mini/data_list_mini.json',
                        help='Data List file for training stored in a JSON format')

    parser.add_argument('--model_dir', '-m', required=False, type=str,
                        default=r'/nfs/masi/nathv/miccai_2020/bottleneck_multi_exp',
                        help='model output directory')

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
    if os.path.exists(model_base_path) is False:
        os.mkdir(model_base_path)

    bottleneck_values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 30, 40, 50]

    ## TODO We loop here for different values of the bottleneck
    for each_bnval in bottleneck_values:

        print('Training Bottleneck model with {} nuerons'.format(each_bnval))

        bn_dir_name = 'bn_' + str(each_bnval)
        bn_model_path = os.path.join(model_base_path, bn_dir_name)
        if os.path.exists(bn_model_path)==False:
            os.mkdir(bn_model_path)


        # Load Json file
        data_list_path = args.data_list
        data_list_path = os.path.normpath(data_list_path)
        all_data = json.load(open(data_list_path))
        tr_data = all_data["train"]
        val_data = all_data["validation"]
        test_data = all_data["test"]

        bottleneck_model = build_sh_patch_resnet_bottleneck(each_bnval)

        patch_crop = [3, 3, 3]

        # Data Generators
        trainGen = nifti_image_generator_patch(tr_data, bs=1000, patch_size=patch_crop)
        validGen = nifti_image_generator_patch(val_data, bs=1000, patch_size=patch_crop)

        # Callbacks for Tensorboard, Saving model with checkpointing
        tensor_board = TensorBoard(log_dir=bn_model_path)

        model_ckpt_name = "weights-improvement-{epoch:02d}-{val_loss:.2f}.hdf5"
        model_ckpt_path = os.path.join(bn_model_path, model_ckpt_name)
        checkpoint = ModelCheckpoint(model_ckpt_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

        callbacks_list = [checkpoint, tensor_board]

        # Fit the DataGenerators
        bottleneck_model.fit_generator(generator=trainGen,
                               validation_data=validGen,
                               steps_per_epoch=100,
                               validation_steps=100,
                               epochs=40,
                               verbose=1,
                               callbacks=callbacks_list)

        print('Training Finished ...')
    return None

if __name__=="__main__":
    main()
