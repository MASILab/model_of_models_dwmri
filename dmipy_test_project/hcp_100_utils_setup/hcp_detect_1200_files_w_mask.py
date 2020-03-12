import os
import nibabel as nib
import numpy as np
import json
import time
import random

def main():

    random.seed(1)
    save_base_path = r'/nfs/masi/nathv/miccai_2020_hcp_100'
    save_base_path = os.path.normpath(save_base_path)

    hcp_base_path = r'/nfs/HCP/data'
    hcp_base_path = os.path.normpath(hcp_base_path)

    hcp_subject_list = os.listdir(hcp_base_path)
    hcp_subject_list_len = len(hcp_subject_list)
    random_idxs = random.sample(range(1, hcp_subject_list_len), 150)

    # Iterate over random subject list, Create the path for picking up data, mask, bvecs, bvals
    all_path_list = []
    for overall_idx, each_subj_idx in enumerate(random_idxs):

        print('Progress {}/{}'.format(overall_idx, len(random_idxs)))
        print('Working on subject ID: {}'.format(hcp_subject_list[each_subj_idx]))
        start_time = time.time()

        t_path_dict = {}
        subject_path = os.path.join(hcp_base_path, hcp_subject_list[each_subj_idx])

        subj_dwi_path = os.path.join(subject_path, 'T1w', 'Diffusion', 'data.nii.gz')
        subj_bval_path = os.path.join(subject_path, 'T1w', 'Diffusion', 'bvals')
        subj_bvec_path = os.path.join(subject_path, 'T1w', 'Diffusion', 'bvecs')
        subj_mask_path = os.path.join(subject_path, 'T1w', 'Diffusion', 'nodif_brain_mask.nii.gz')
        subj_t1_path = os.path.join(subject_path, 'T1w', 'T1w', 'T1w_acpc_dc_restore_1.25.nii.gz')

        # Sanity check existence of data
        if os.path.exists(subj_dwi_path)==False:
            print('Data file does not exist for {}'.format(hcp_subject_list[each_subj_idx]))
            continue

        if os.path.exists(subj_bval_path)==False:
            print('Bval file does not exist for {}'.format(hcp_subject_list[each_subj_idx]))
            continue

        if os.path.exists(subj_bvec_path)==False:
            print('Bvec file does not exist for {}'.format(hcp_subject_list[each_subj_idx]))
            continue

        if os.path.exists(subj_mask_path)==False:
            print('Mask file does not exist for {}'.format(hcp_subject_list[each_subj_idx]))
            continue

        if os.path.exists(subj_t1_path)==False:
            print('T1 file does not exist for {}'.format(hcp_subject_list[each_subj_idx]))
            continue


        # Read the data
        bvals = np.loadtxt(subj_bval_path)
        bvecs = np.loadtxt(subj_bvec_path)

        if len(bvals)<270:
            print('Less than expected gradient volumes for subject: {}, gradient vols: {}'.format(hcp_subject_list[each_subj_idx], len(bvals)))
            continue

        dwi_data_obj = nib.load(subj_dwi_path)
        dwi_data = dwi_data_obj.get_fdata()
        dwi_data_obj.uncache()

        t1_data_obj = nib.load(subj_t1_path)
        t1_data = t1_data_obj.get_fdata()
        t1_data_obj.uncache()

        mask_data_obj = nib.load(subj_mask_path)
        mask_data = mask_data_obj.get_fdata()
        mask_data_obj.uncache()

        # Extract indices from bvalues where 1000 and 0 are present
        idxs_1k = [i for i in range(len(bvals)) if (bvals[i] > 900 and bvals[i] < 1100)]
        idxs_b0 = [i for i in range(len(bvals)) if (bvals[i] < 50)]

        # Extract the b1000 data
        t1_dims = t1_data.shape
        dwi_dims = dwi_data.shape

        print('Data Loaded ...')
        #assert (t1_dims == (dwi_dims[0], dwi_dims[1], dwi_dims[2]))

        b1k_data = np.zeros((dwi_dims[0], dwi_dims[1], dwi_dims[2], len(idxs_1k)+len(idxs_b0)))
        b1k_bvals = np.zeros((len(idxs_1k)+len(idxs_b0), 1))
        b1k_bvecs = np.zeros((3, len(idxs_1k)+len(idxs_b0)))

        b1k_data[:, :, :, 0:len(idxs_b0)] = dwi_data[:, :, :, idxs_b0]
        b1k_data[:, :, :, len(idxs_b0):(len(idxs_b0) + len(idxs_1k))] = dwi_data[:, :, :, idxs_1k]

        b1k_bvals[0:len(idxs_b0), 0] = bvals[idxs_b0]
        b1k_bvals[len(idxs_b0):(len(idxs_b0)+len(idxs_1k)), 0] = bvals[idxs_1k]

        b1k_bvecs[:, 0:len(idxs_b0)] = bvecs[:, idxs_b0]
        b1k_bvecs[:, len(idxs_b0):(len(idxs_b0) + len(idxs_1k))] = bvecs[:, idxs_1k]

        # Saving Data
        subj_save_path = os.path.join(save_base_path, hcp_subject_list[each_subj_idx])
        if os.path.exists(subj_save_path) == False:
            os.mkdir(subj_save_path)

        dwi_1k_img = nib.Nifti1Image(b1k_data, dwi_data_obj.affine, dwi_data_obj.header)
        subj_save_dwi_1k_path = os.path.join(subj_save_path, 'dwi_1k_data.nii.gz')
        nib.save(dwi_1k_img, subj_save_dwi_1k_path)

        mask_img = nib.Nifti1Image(mask_data, mask_data_obj.affine, mask_data_obj.header)
        subj_save_mask_path = os.path.join(subj_save_path, 'mask_data.nii.gz')
        nib.save(mask_img, subj_save_mask_path)

        t1_img = nib.Nifti1Image(t1_data, t1_data_obj.affine, t1_data_obj.header)
        subj_save_t1_path = os.path.join(subj_save_path, 't1_data.nii.gz')
        nib.save(t1_img, subj_save_t1_path)

        subj_save_1k_bvals_path = os.path.join(subj_save_path, 'bvals_1k')
        np.savetxt(subj_save_1k_bvals_path, b1k_bvals)

        subj_save_1k_bvecs_path = os.path.join(subj_save_path, 'bvecs_1k')
        np.savetxt(subj_save_1k_bvecs_path, b1k_bvecs)

        end_time = time.time()
        print('Total time taken: {}'.format(end_time-start_time))

        # Grab ID from path to header
        # f_name = path_to_nifti_header.split('\\')
        print('Debug here')







    return None

if __name__=="__main__":
    main()