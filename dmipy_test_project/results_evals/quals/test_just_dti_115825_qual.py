import os
import numpy as np
import nibabel as nib
import json
import time
import matplotlib.pyplot as plt
from scipy.ndimage import rotate

def load_nifty(path_to_file, data_type):
    start_time = time.time()
    nifti_data = nib.load(path_to_file)
    nifti_img = nifti_data.get_fdata(dtype=data_type)
    nifti_data.uncache()
    end_time = time.time()
    print('\n Time Take to Read {}'.format(end_time - start_time))
    return nifti_img

def main():
    # Ground Truth Paths
    json_path = r'/nfs/masi/nathv/miccai_2020/micro_methods_hcp_mini/data_list_dti.json'
    json_path = os.path.normpath(json_path)

    all_data = json.load(open(json_path))
    #tr_data = all_data["train"]
    #val_data = all_data["validation"]
    test_data = all_data["test"]
    gt_paths_dict = test_data['output']
    method_list = ['DTI']

    # Predicted Data Paths
    base_pred_path = r'/nfs/masi/nathv/miccai_2020/double_bottleneck_just_dti_test/bn_50/predicted_volumes'
    base_pred_path = os.path.normpath(base_pred_path)

    # TODO Slice number is stored here
    slice_num = 72

    # Load the mask and make it boolean
    mask_data = load_nifty(test_data['mask'], data_type='float32')
    mask_bool = np.array(mask_data, dtype='bool')

    # DTI Analysis
    pred_dti_path = os.path.join(base_pred_path, 'dti.nii.gz')
    pred_dti = load_nifty(pred_dti_path, 'float32')

    plt.figure(1, figsize=(8, 12))
    metric_nums = len(gt_paths_dict['DTI'])
    plot_counter = 1
    for idx, each_vol_path in enumerate(gt_paths_dict['DTI']):

        gt_vol = load_nifty(each_vol_path, 'float32')
        plt.subplot(3, metric_nums, plot_counter)
        plt.imshow(rotate(np.squeeze(gt_vol[:, :, slice_num]), 90))
        plt.axis('equal')
        plt.colorbar()
        plt.clim(0, 1)
        plt.title('GT')


        plt.subplot(3, metric_nums, plot_counter + metric_nums)
        plt.imshow(rotate(np.squeeze(pred_dti[:, :, slice_num, idx]), 90))
        plt.axis('equal')
        plt.colorbar()
        plt.clim(0, 1)
        plt.title('Predicted')

        diff_img = np.abs(np.squeeze(gt_vol[:, :, slice_num]) - np.squeeze(pred_dti[:, :, slice_num, idx]))
        plt.subplot(3, metric_nums, plot_counter + metric_nums*2)
        plt.imshow(rotate(diff_img, 90))
        plt.axis('equal')
        plt.colorbar()
        plt.clim(0, 0.1)

        # Overall Volume Error
        gt_mask = gt_vol[mask_bool]
        pred_mask = pred_dti[:,:,:,idx]
        pred_mask = pred_mask[mask_bool]

        masked_error_mean = np.mean(np.abs(gt_mask - pred_mask))
        plt.title('Abs Diff On Volume {}'.format(masked_error_mean))

        plot_counter = plot_counter + 1
    plt.show()
    plt.clf()
    plt.close(1)

    print('Debug here')
    return None

if __name__=="__main__":
    main()