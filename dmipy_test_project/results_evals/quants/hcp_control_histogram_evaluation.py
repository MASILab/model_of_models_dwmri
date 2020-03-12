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
    json_path = r'/nfs/masi/nathv/miccai_2020/micro_methods_hcp_mini/data_list_compart_dti.json'
    json_path = os.path.normpath(json_path)

    all_data = json.load(open(json_path))
    #tr_data = all_data["train"]
    #val_data = all_data["validation"]
    test_data = all_data["test"]
    gt_paths_dict = test_data['output']
    method_list = ['BS_2003', 'IVIM', 'MC_SMT', 'NODDI_WATSON', 'DTI']

    # Predicted Data Paths
    base_pred_path = r'/nfs/masi/nathv/miccai_2020/bottleneck_compart_dti_test/bn_15/predicted_volumes'
    base_pred_path = os.path.normpath(base_pred_path)

    # Load the Mask Data
    mask_data = load_nifty(test_data['mask'], data_type='float32')
    mask_bool = np.array(mask_data, dtype='bool')

    # TODO Slice number is stored here
    slice_num = 72

    # DTI Analysis
    pred_dti_path = os.path.join(base_pred_path, 'dti.nii.gz')
    pred_dti = load_nifty(pred_dti_path, 'float32')

    plt.figure(1, figsize=(8, 12))
    metric_nums = len(gt_paths_dict['DTI'])
    plot_counter = 1
    for idx, each_vol_path in enumerate(gt_paths_dict['DTI']):

        gt_vol = load_nifty(each_vol_path, 'float32')
        plt.subplot(1, metric_nums, plot_counter)
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
        plt.title('Abs Difference')

        plot_counter = plot_counter + 1
    plt.clf()
    plt.close(1)

    # IVIM Analysis
    pred_dti_path = os.path.join(base_pred_path, 'ivim.nii.gz')
    pred_dti = load_nifty(pred_dti_path, 'float32')

    plt.figure(1, figsize=(12, 12))
    metric_nums = len(gt_paths_dict['IVIM'])
    plot_counter = 1
    for idx, each_vol_path in enumerate(gt_paths_dict['IVIM']):

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
        plt.title('Abs Difference')

        plot_counter = plot_counter + 1
    plt.clf()
    plt.close(1)

    # MC SMT Analysis
    pred_dti_path = os.path.join(base_pred_path, 'mc_smt.nii.gz')
    pred_dti = load_nifty(pred_dti_path, 'float32')

    plt.figure(1, figsize=(8, 12))
    metric_nums = len(gt_paths_dict['MC_SMT'])
    plot_counter = 1
    for idx, each_vol_path in enumerate(gt_paths_dict['MC_SMT']):

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
        plt.clim(0, 0.2)
        plt.title('Abs Difference')

        plot_counter = plot_counter + 1
    plt.clf()
    plt.close(1)

    # Ball Stick Analysis
    pred_dti_path = os.path.join(base_pred_path, 'ball_stick.nii.gz')
    pred_dti = load_nifty(pred_dti_path, 'float32')

    plt.figure(1, figsize=(20, 12))
    metric_nums = len(gt_paths_dict['BS_2003'])
    plot_counter = 1
    for idx, each_vol_path in enumerate(gt_paths_dict['BS_2003']):

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
        plt.clim(0, 0.2)
        plt.title('Abs Difference')

        plot_counter = plot_counter + 1

    plt.clf()
    plt.close(1)

    # NODDI Analysis
    pred_dti_path = os.path.join(base_pred_path, 'noddi.nii.gz')
    pred_dti = load_nifty(pred_dti_path, 'float32')

    plt.figure(1, figsize=(18, 12))
    metric_nums = len(gt_paths_dict['NODDI_WATSON'])
    plot_counter = 1
    for idx, each_vol_path in enumerate(gt_paths_dict['NODDI_WATSON']):

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
        plt.clim(0, 0.2)
        plt.title('Abs Difference')

        plot_counter = plot_counter + 1

    plt.show()
    print('Debug here')
    return None

if __name__=="__main__":
    main()