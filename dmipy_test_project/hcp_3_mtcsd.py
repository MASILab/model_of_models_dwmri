import os
import numpy as np
import nibabel as nib
import time
import matplotlib.pyplot as plt

# Dmipy imports
from dmipy.core.acquisition_scheme import acquisition_scheme_from_bvalues

# MT-CSD Imports
from dmipy.tissue_response.three_tissue_response import three_tissue_response_dhollander16
from dmipy.core.modeling_framework import MultiCompartmentSphericalHarmonicsModel

def main():
    # Plot Save Path
    base_plot_path = r'/nfs/masi/nathv/py_src_code_2020/dmipy_model_pictures'
    base_plot_path = os.path.normpath(base_plot_path)

    # Method Saving Paths
    base_save_path = r'/nfs/masi/nathv/miccai_2020/micro_methods_hcp_mini'
    base_save_path = os.path.normpath(base_save_path)

    # Create base saving path for Method
    # TODO The Method name can be made an argument later on
    method_name = 'MT_CSD'

    # Base HCP Data Path
    base_data_path = r'/nfs/HCP/data'
    base_data_path = os.path.normpath(base_data_path)


    # Subject ID's list
    subj_ID_List = ['115017', '114823', '116726', '118225', '115825', '125525']
    # TODO When needed loop here over the ID list
    for subj_ID in subj_ID_List:
        # Subject Save Path
        subj_save_path = os.path.join(base_save_path, subj_ID)
        if os.path.exists(subj_save_path)==False:
            os.mkdir(subj_save_path)

        # TODO For later the subject data, bval and bvec reading part can be put inside a function
        subj_data_path = os.path.join(base_data_path, subj_ID, 'T1w', 'Diffusion')

        # Read the Nifti file, bvals and bvecs
        subj_bvals = np.loadtxt(os.path.join(subj_data_path, 'bvals'))
        subj_bvecs = np.loadtxt(os.path.join(subj_data_path, 'bvecs'))

        all_bvals = subj_bvals * 1e6
        all_bvecs = np.transpose(subj_bvecs)

        subj_Acq_Scheme = acquisition_scheme_from_bvalues(all_bvals, all_bvecs)
        print(subj_Acq_Scheme.print_acquisition_info)

        print('Loading the Nifti Data ...')
        data_start_time = time.time()

        subj_babel_object = nib.load(os.path.join(subj_data_path, 'data.nii.gz'))
        subj_data = subj_babel_object.get_fdata()
        axial_slice_data = subj_data[:, :, 30:32, :]

        data_end_time = time.time()
        data_time = np.int(np.round(data_end_time - data_start_time))

        print('Data Loaded ... Time Taken: {}'.format(data_end_time - data_start_time))
        print('The Data Dimensions are: {}'.format(subj_data.shape))

        #### MT-CSD Begin ####

        S0_tissue_responses, tissue_response_models, selection_map = three_tissue_response_dhollander16(
            subj_Acq_Scheme, subj_data, wm_algorithm='tournier13',
            wm_N_candidate_voxels=10, gm_perc=0.2, csf_perc=0.4)
        TR2_wm, TR1_gm, TR1_csf = tissue_response_models
        S0_wm, S0_gm, S0_csf = S0_tissue_responses

        mt_csd_mod = MultiCompartmentSphericalHarmonicsModel(
            models=tissue_response_models,
            S0_tissue_responses=S0_tissue_responses)

        fit_args = {'acquisition_scheme': subj_Acq_Scheme, 'data': subj_data, 'mask': subj_data[..., 0] > 0}

        mt_csd_fits = []
        for fit_S0_response in [True, False]:
            mt_csd_fits.append(mt_csd_mod.fit(fit_S0_response=fit_S0_response, **fit_args))

        # Get List of Estimated Parameter Names
        para_Names_list = mt_csd_mod.parameter_names

        print('Fitting the MT-CSD Model ...')
        fit_start_time = time.time()
        #mcdmi_fit = mcdmi_mod.fit(subj_Acq_Scheme, subj_data, mask=subj_data[..., 0] > 0)
        fit_end_time = time.time()
        print('Model Fitting Completed ... Time Taken to fit: {}'.format(fit_end_time - fit_start_time))
        fit_time = np.int(np.round(fit_end_time - fit_start_time))

        fitted_parameters = mt_csd_fits.fitted_parameters

        ### Nifti Saving Part
        # Create a directory per subject
        subj_method_save_path = os.path.join(subj_save_path, method_name)
        if os.path.exists(subj_method_save_path)==False:
            os.mkdir(subj_method_save_path)

        # Retrieve the affine from already Read Nifti file to form the header
        affine = subj_babel_object.affine

        # Loop over fitted parameters name list
        for each_fitted_parameter in para_Names_list:
            new_img = nib.Nifti1Image(fitted_parameters[each_fitted_parameter], affine)

            # Form the file path
            f_name = each_fitted_parameter + '.nii.gz'
            param_file_path = os.path.join(subj_method_save_path, f_name)

            nib.save(new_img, param_file_path)

    return None


if __name__ == '__main__':
    main()