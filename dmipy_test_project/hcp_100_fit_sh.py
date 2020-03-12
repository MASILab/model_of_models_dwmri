import os
import argparse

def main():

    #Argparse Stuff
    parser = argparse.ArgumentParser(description='start_idx')
    parser.add_argument('--start_idx', type=int, default=10)

    parser.add_argument('--end_idx', type=int, default=20)
    args = parser.parse_args()

    start_idx = args.start_idx
    end_idx = args.end_idx

    base_path = r'/nfs/masi/nathv/miccai_2020_hcp_100'
    base_path = os.path.normpath(base_path)

    dir_list = os.listdir(base_path)

    dir_list = dir_list[start_idx:end_idx]

    for each_subj in dir_list:
        print('Working on subject number {}'.format(each_subj))
        # Construct the Subject Path
        subj_path = os.path.join(base_path, each_subj)

        # Construct the mrtrix3 command
        mrtrix_cmd = 'amp2sh -normalise -lmax 8 '
        bvec_path = os.path.join(subj_path, 'bvecs_1k')
        bval_path = os.path.join(subj_path, 'bvals_1k')
        data_path = os.path.join(subj_path, 'dwi_1k_data.nii.gz')

        data_mif_path = os.path.join(subj_path, 'dwi.mif')
        data_denoised_mif = os.path.join(subj_path, 'dwi_denoised.mif')
        output_mif_path = os.path.join(subj_path, 'sh.mif')
        output_path = os.path.join(subj_path, 'sh_dwi_1k.nii.gz')

        print('Converting the DWI Data to MIF')
        mrtrix_mrconvert_cmd = r'mrconvert'
        mrtrix_mrconvert_cmd = mrtrix_mrconvert_cmd + ' ' + '-fslgrad' + ' ' + bvec_path + ' ' + bval_path + ' ' + '-datatype float32' + ' ' + data_path + ' ' + data_mif_path
        os.system(mrtrix_mrconvert_cmd)

        # Pre-processing the data with Denoising and Unringing.
        print('Running Denoising and unringing')
        # r'dwidenoise ${ANALYSISDIR}/${subj}/DWI.mif - | mrdegibbs - ${ANALYSISDIR}/${subj}/DWI_denoise_unring.mif'
        mrtrix_denoise_cmd = r'dwidenoise'
        mrtrix_denoise_cmd = mrtrix_denoise_cmd + ' ' + data_mif_path + ' -  | mrdegibbs - ' + data_denoised_mif
        os.system(mrtrix_denoise_cmd)

        # Run Amp2SH
        mrtrix_cmd = mrtrix_cmd + data_denoised_mif + ' ' + output_mif_path + ' ' + '-force'
        os.system(mrtrix_cmd)

        # Run mrconvert again
        mrtrix_convert_nifti_cmd = 'mrconvert'
        mrtrix_convert_nifti_cmd = mrtrix_convert_nifti_cmd + ' ' + output_mif_path + ' ' + output_path + ' -force'

        # Delete unnecessary files
        print('Removing Temporary Generated files ...')
        os.system('rm ' + data_mif_path)
        os.system('rm ' + data_denoised_mif)
        os.system('rm ' + output_mif_path)

        print('Done.')
    return None

if __name__=="__main__":
    main()
