import os
import numpy as np
import random
import argparse
import time
import json
import pickle
import matplotlib.pyplot as plt

from models.bottleneck_patch_models import build_patch_double_bottleneck

def main():

    bn_base_path = r'/nfs/masi/nathv/miccai_2020/double_bottleneck_compart_dti/'
    bn_base_path = os.path.normpath(bn_base_path)

    plt.figure(1, figsize=(14,7))

    for bn_values in range(1,11):
        bn_weights_path = os.path.join(bn_base_path, 'bn_{}'.format(bn_values), 'weights-improvement.hdf5')

        # Build the model with the bottleneck number
        bn_model = build_patch_double_bottleneck(bn_values)

        # Load the weights
        bn_model.load_weights(bn_weights_path)

        grabbed_weights = bn_model.get_weights()

        # TODO We want to analyze the layer number 40
        bottled_features = grabbed_weights[40]

        plt.subplot(2, 5, bn_values)
        plt.imshow(bottled_features)
        plt.colorbar()
        plt.title('{} x {}'.format(bn_values, bn_values))

    plt.show()
    print('Debug here')
    return None

if __name__=="__main__":
    main()