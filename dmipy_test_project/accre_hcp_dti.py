import os
import numpy as np
import argparse

from dipy.io.image import load_nifti, save_nifti
from dipy.io.gradients import read_bvals_bvecs
from dipy.core.gradients import gradient_table
from dipy.segment.mask import median_otsu
from dipy.reconst.dti import fractional_anisotropy, color_fa

import dipy.reconst.dti as dti
import nibabel as nib

def main():
    #Argparse Stuff
    parser = argparse.ArgumentParser(description='subject_id')
    parser.add_argument('--subject_id', type=str, default='135124')
    args = parser.parse_args()

    # Method Saving Paths
    # TODO KARTHIK
    base_save_path = r'/root/hcp_results'
    base_save_path = os.path.normpath(base_save_path)
    if os.path.exists(base_save_path)==False:
        os.mkdir(base_save_path)

    # Create base saving path for Method
    # TODO The Method name can be made an argument later on
    method_name = 'DTI'

    # Base HCP Data Path
    # TODO KARTHIK This is where we hard set HCP's Data Path
    base_data_path = r'/root/local_mount/data'
    base_data_path = os.path.normpath(base_data_path)


    # Subject ID's list
    #subj_ID_List = ['115017', '114823', '116726', '118225', '115825', '125525']
    #subj_ID_List = ['100610', '102311', '102816', '104416', '105923', '108323', '109123', '111312', '111514']

    # Subject ID
    subj_ID = args.subject_id

    # Subject Save Path

    print('Working on subject ID: {}'.format(subj_ID))
    subj_save_path = os.path.join(base_save_path, subj_ID)
    if os.path.exists(subj_save_path)==False:
        os.mkdir(subj_save_path)

    # TODO For later the subject data, bval and bvec reading part can be put inside a function
    subj_data_path = os.path.join(base_data_path, subj_ID, 'T1w', 'Diffusion')

    # Read the Nifti file, bvals and bvecs
    subj_bvals = os.path.join(subj_data_path, 'bvals')
    subj_bvecs = os.path.join(subj_data_path, 'bvecs')

    subj_babel_object = nib.load(os.path.join(subj_data_path, 'data.nii.gz'))
    subj_data = subj_babel_object.get_fdata()

    # Load the mask
    mask_babel_object = nib.load(os.path.join(subj_data_path, 'nodif_brain_mask.nii.gz'))
    mask_data = mask_babel_object.get_data()


    # Prepping Bvals, Bvecs and forming the gradient table using dipy
    bvals, bvecs = read_bvals_bvecs(subj_bvals, subj_bvecs)
    gtab = gradient_table(bvals, bvecs)
    print('Gradient Table formed ...')

    #maskdata, mask = median_otsu(subj_data, vol_idx=range(10, 50), median_radius=3,
    #                            numpass=1, autocrop=True, dilate=2)
    #print('maskdata.shape (%d, %d, %d, %d)' % maskdata.shape)

    # Form the tensor model
    tenmodel = dti.TensorModel(gtab)

    ## Loop over the data and mask
    data_dims = subj_data.shape

    fa_vol = np.zeros((data_dims[0], data_dims[1], data_dims[2]))
    md_vol = np.zeros((data_dims[0], data_dims[1], data_dims[2]))

    for x in range(0,data_dims[0]):
        print(x)
        for y in range(0, data_dims[1]):
            for z in range(0, data_dims[2]):

                if mask_data[x,y,z] == 1:
                    # Fit the tensor and calculate FA and MD
                    tenfit = tenmodel.fit(subj_data[x, y, z, :])

                    # Eval FA and MD and assign to empty vols
                    FA = fractional_anisotropy(tenfit.evals)
                    MD1 = dti.mean_diffusivity(tenfit.evals)

                    fa_vol[x,y,z] = FA
                    md_vol[x,y,z] = MD1

    ### Nifti Saving Part
    # Create a directory per subject
    subj_method_save_path = os.path.join(subj_save_path, method_name)
    if os.path.exists(subj_method_save_path)==False:
        os.mkdir(subj_method_save_path)

    # Retrieve the affine from already Read Nifti file to form the header
    affine = subj_babel_object.affine

    # Form the file path
    fa_file_path = os.path.join(subj_method_save_path, 'tensor_fa.nii.gz')
    md_file_path = os.path.join(subj_method_save_path, 'tensor_md.nii.gz')

    print('Computing anisotropy measures (FA, MD, RGB)')

    save_nifti(fa_file_path, fa_vol.astype(np.float32), affine)

    save_nifti(md_file_path, md_vol.astype(np.float32), affine)

    return None

if __name__ == '__main__':
    main()