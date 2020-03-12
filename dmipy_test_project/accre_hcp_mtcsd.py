import os
import csv
import numpy as np
import time
import argparse

def main():

    #Argparse Stuff
    parser = argparse.ArgumentParser(description='subject_id')
    parser.add_argument('--subject_id', type=str, default='135124')
    args = parser.parse_args()

    # Read the CSV list
    #csv_path = r'/nfs/masi/nathv/py_src_code_2020/dmipy_test_project/hcp_102.csv'
    #csv_path = os.path.normpath(csv_path)
    #csv_data = np.loadtxt(csv_path)
    #csv_clean = []
    '''
    # There is an additional . stored in the list, clean it
    for idx, each_row in enumerate(csv_data):
        temp_row = each_row
        temp_row = str(int(temp_row))
        #csv_data[idx] = temp_row[0:6]
        csv_clean.append(temp_row[0:6])

    print(csv_clean)
    '''

    # Method Saving Paths
    # TODO KARTHIK
    hcp_save_path = r'/OUTPUTS/hcp_results'
    hcp_save_path = os.path.normpath(hcp_save_path)
    if os.path.exists(hcp_save_path)==False:
        os.mkdir(hcp_save_path)

    # Saving Directory setup
   # hcp_save_path = r'/nfs/masi/nathv/miccai_2020_hcp_100'

    # Base HCP Data Path
    # TODO KARTHIK This is where we hard set HCP's Data Path
    hcp_base_path = r'/INPUTS/data'
    hcp_base_path = os.path.normpath(hcp_base_path)

    # The Data is clean now, Setup HCP base path
    #hcp_base_path = r'/nfs/HCP/data'
    #hcp_base_path = os.path.normpath(hcp_base_path)

    # Loop over the subject list
    #for idx, each_subject in enumerate(csv_clean):

    #print('Progress: {} / {}'.format(idx, len(csv_clean)))

    # Subject ID
    each_subject = args.subject_id

    print('Working on Subject ID: {}'.format(each_subject))

    start_time = time.time()

    hcp_dwi_subject_path = os.path.join(hcp_base_path, each_subject, 'T1w', 'Diffusion')
    hcp_t1_subject_path = os.path.join(hcp_base_path, each_subject, 'T1w', 'T1w', 'T1w_acpc_dc_restore_1.25.nii.gz')

    hcp_dwi_data = os.path.join(hcp_dwi_subject_path, 'data.nii.gz')
    hcp_mask_data = os.path.join(hcp_dwi_subject_path, 'nodif_brain_mask.nii.gz')
    hcp_bvals = os.path.join(hcp_dwi_subject_path, 'bvals')
    hcp_bvecs = os.path.join(hcp_dwi_subject_path, 'bvecs')

    hcp_subject_save_path = os.path.join(hcp_save_path, each_subject)
    hcp_subject_dwi_mif_path = os.path.join(hcp_subject_save_path, 'DWI.mif')
    hcp_subject_dwi_denoised_mif_path = os.path.join(hcp_subject_save_path, 'DWI_denoised.mif')
    hcp_subject_t1_5tt_mif_path = os.path.join(hcp_subject_save_path, '5TT.mif')

    hcp_subject_voxels_mif_path = os.path.join(hcp_subject_save_path, 'voxels.mif')
    hcp_subject_resp_wm_path = os.path.join(hcp_subject_save_path, 'rf_wm.txt')
    hcp_subject_wm_mif_path = os.path.join(hcp_subject_save_path, 'wm_fods.mif')
    hcp_subject_resp_gm_path = os.path.join(hcp_subject_save_path, 'rf_gm.txt')
    hcp_subject_gm_mif_path = os.path.join(hcp_subject_save_path, 'gm.mif')
    hcp_subject_resp_csf_path = os.path.join(hcp_subject_save_path, 'rf_csf.txt')
    hcp_subject_csf_mif_path = os.path.join(hcp_subject_save_path, 'csf.mif')
    hcp_subject_tissue_volfrac_mif_path = os.path.join(hcp_subject_save_path, 'tissueVolFrac.mif')

    # Create a directory for MT-CSD
    hcp_subject_mtcsd_dir = os.path.join(hcp_subject_save_path, 'MT_CSD')
    if os.path.exists(hcp_subject_mtcsd_dir)==False:
        os.mkdir(hcp_subject_mtcsd_dir)

    mt_csd_wm_nifti_path = os.path.join(hcp_subject_mtcsd_dir, 'wm_fods.nii.gz')
    mt_csd_volfrac_nifti_path = os.path.join(hcp_subject_mtcsd_dir, 'vol_frac.nii.gz')


    # Converting Data to Mrtrix's MIF format,  -strides 0,0,0,1
    print('Converting the DWI Data to MIF')
    mrtrix_mrconvert_cmd = r'mrconvert'
    mrtrix_mrconvert_cmd = mrtrix_mrconvert_cmd + ' ' + '-fslgrad' + ' ' + hcp_bvecs + ' ' + hcp_bvals + ' ' + '-datatype float32' + ' ' + hcp_dwi_data + ' ' + hcp_subject_dwi_mif_path
    os.system(mrtrix_mrconvert_cmd)

    # Pre-processing the data with Denoising and Unringing.
    print('Running Denoising and unringing')
    # r'dwidenoise ${ANALYSISDIR}/${subj}/DWI.mif - | mrdegibbs - ${ANALYSISDIR}/${subj}/DWI_denoise_unring.mif'
    mrtrix_denoise_cmd = r'dwidenoise'
    mrtrix_denoise_cmd = mrtrix_denoise_cmd + ' ' + hcp_subject_dwi_mif_path + ' -  | mrdegibbs - ' + hcp_subject_dwi_denoised_mif_path
    os.system(mrtrix_denoise_cmd)

    # Running 5TT algorithm on the T1 image
    #5ttgen fsl ${HCPDATADIR}/${subj}/T1w/T1w_acpc_dc_restore_1.25.nii.gz ${ANALYSISDIR}/${subj}/5TT.mif
    print('Running the 5TT Algorithm on')
    mrtrix_5tt_cmd = '5ttgen fsl'
    mrtrix_5tt_cmd = mrtrix_5tt_cmd + ' ' + hcp_t1_subject_path + ' ' + hcp_subject_t1_5tt_mif_path
    os.system(mrtrix_5tt_cmd)

    # Estimate the Multi-Tissue CSD Response Functions
    # dwi2response msmt_5tt ${ANALYSISDIR}/${subj}/DWI_denoise_unring.mif ${ANALYSISDIR}/${subj}/5TT.mif ${ANALYSISDIR}/${subj}/rf_wm.txt ${ANALYSISDIR}/${subj}/rf_gm.txt ${ANALYSISDIR}/${subj}/rf_csf.txt -voxels ${ANALYSISDIR}/${subj}/rf_voxels.mif
    print('Estimating the Response Functions for MT-CSD')
    mrtrix_mtcsd_resp_cmd = 'dwi2response msmt_5tt'
    mrtrix_mtcsd_resp_cmd = mrtrix_mtcsd_resp_cmd + ' ' + hcp_subject_dwi_denoised_mif_path + ' ' + hcp_subject_t1_5tt_mif_path + ' ' + hcp_subject_resp_wm_path + ' ' + hcp_subject_resp_gm_path + ' ' + hcp_subject_resp_csf_path + ' ' + '-voxels' + ' ' + hcp_subject_voxels_mif_path
    os.system(mrtrix_mtcsd_resp_cmd)

    # Estimate the Fiber ODF of MT-CSD
    # dwi2fod msmt_csd ${ANALYSISDIR}/${subj}/DWI_denoise_unring.mif ${ANALYSISDIR}/${subj}/rf_wm.txt ${ANALYSISDIR}/${subj}/wm_fods.mif ${ANALYSISDIR}/${subj}/rf_gm.txt ${ANALYSISDIR}/${subj}/gm.mif ${ANALYSISDIR}/${subj}/rf_csf.txt ${ANALYSISDIR}/${subj}/csf.mif -mask ${HCPDATADIR}/${subj}/T1w/Diffusion/nodif_brain_mask.nii.gz
    print('Estimating MT-CSD FODF ')
    mrtrix_mtcsd_fodf_cmd = 'dwi2fod msmt_csd'
    mrtrix_mtcsd_fodf_cmd = mrtrix_mtcsd_fodf_cmd + ' ' + hcp_subject_dwi_denoised_mif_path + ' ' + hcp_subject_resp_wm_path + ' ' + hcp_subject_wm_mif_path + ' ' + hcp_subject_resp_gm_path + ' ' + hcp_subject_gm_mif_path + ' ' + hcp_subject_resp_csf_path + ' ' + hcp_subject_csf_mif_path + ' ' + '-mask ' + hcp_mask_data
    os.system(mrtrix_mtcsd_fodf_cmd)


    # Use MRConvert to go back to Nifti file and save the WM FODS and Vol Fractions
    # mrconvert ${ANALYSISDIR}/${subj}/wm_fods.mif - -coord 3 0 | mrcat ${ANALYSISDIR}/${subj}/csf.mif ${ANALYSISDIR}/${subj}/gm.mif - ${ANALYSISDIR}/${subj}/tissueVolFrac.mif -axis 3
    print('Converting all back to Nifti')
    mrtrix_nifti_cmd = 'mrconvert'
    mrtrix_nifti_cmd = mrtrix_nifti_cmd + ' ' + hcp_subject_wm_mif_path + ' - ' + '-coord 3 0' + ' | mrcat' + ' ' + hcp_subject_csf_mif_path + ' ' + hcp_subject_gm_mif_path + ' ' + '-' + ' ' + hcp_subject_tissue_volfrac_mif_path + ' -axis 3'
    os.system(mrtrix_nifti_cmd)

    mrtrix_nifti_cmd = 'mrconvert'
    mrtrix_nifti_cmd = mrtrix_nifti_cmd + ' ' + hcp_subject_wm_mif_path + ' ' + mt_csd_wm_nifti_path
    os.system(mrtrix_nifti_cmd)

    mrtrix_nifti_cmd = 'mrconvert'
    mrtrix_nifti_cmd = mrtrix_nifti_cmd + ' ' + hcp_subject_tissue_volfrac_mif_path + ' ' + mt_csd_volfrac_nifti_path
    os.system(mrtrix_nifti_cmd)

    end_time = time.time()
    print('Total Time Taken: {}'.format(end_time - start_time))
    print('All Done')

    return None

if __name__=="__main__":
    main()