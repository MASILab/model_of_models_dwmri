import os
import numpy as np
import nibabel as nib
import time
import matplotlib.pyplot as plt
import argparse

# Dmipy imports
from dmipy.core.acquisition_scheme import acquisition_scheme_from_bvalues

# MC-SMT Imports
from dmipy.signal_models import cylinder_models, gaussian_models
from dmipy.distributions.distribute_models import BundleModel
from dmipy.core import modeling_framework

def main():
    #Argparse Stuff
    parser = argparse.ArgumentParser(description='subject_id')
    parser.add_argument('--subject_id', type=str, default='135124')
    args = parser.parse_args()

    # Plot Save Path
    base_plot_path = r'/nfs/masi/nathv/py_src_code_2020/dmipy_model_pictures'
    base_plot_path = os.path.normpath(base_plot_path)

    # Method Saving Paths
    # TODO KARTHIK
    base_save_path = r'/root/hcp_results'
    base_save_path = os.path.normpath(base_save_path)
    if os.path.exists(base_save_path)==False:
        os.mkdir(base_save_path)

    # Create base saving path for Method
    # TODO The Method name can be made an argument later on
    method_name = 'MC_SMT'

    # Base HCP Data Path
    # TODO KARTHIK This is where we hard set HCP's Data Path
    base_data_path = r'/root/local_mount/data'
    base_data_path = os.path.normpath(base_data_path)
    #base_data_path = args.input_path

    # Subject ID
    subj_ID = args.subject_id

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

    #### MC-SMT Begin ####
    stick = cylinder_models.C1Stick()
    zeppelin = gaussian_models.G2Zeppelin()

    bundle = BundleModel([stick, zeppelin])

    # Model Paramter Constraints
    bundle.set_tortuous_parameter('G2Zeppelin_1_lambda_perp',
                                  'C1Stick_1_lambda_par', 'partial_volume_0')
    bundle.set_equal_parameter('G2Zeppelin_1_lambda_par', 'C1Stick_1_lambda_par')

    mcdmi_mod = modeling_framework.MultiCompartmentSphericalMeanModel(
        models=[bundle])

    # Get List of Estimated Parameter Names
    para_Names_list = mcdmi_mod.parameter_names

    print('Fitting the MC-SMT Model ...')
    fit_start_time = time.time()
    mcdmi_fit = mcdmi_mod.fit(subj_Acq_Scheme,
                              subj_data,
                              mask=subj_data[..., 0] > 0,
                              use_parallel_processing=True,
                              number_of_processors=32)
    
    fit_end_time = time.time()
    print('Model Fitting Completed ... Time Taken to fit: {}'.format(fit_end_time - fit_start_time))
    fit_time = np.int(np.round(fit_end_time - fit_start_time))

    fitted_parameters = mcdmi_fit.fitted_parameters

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