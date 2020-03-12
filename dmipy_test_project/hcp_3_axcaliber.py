import os
import numpy as np
import nibabel as nib
import time
import matplotlib.pyplot as plt

# Dmipy imports
from dmipy.core.acquisition_scheme import acquisition_scheme_from_bvalues

# Verdict Imports
from dmipy.signal_models import cylinder_models, gaussian_models
from dmipy.distributions import distribute_models
from dmipy.core import modeling_framework

def main():
    # Plot Save Path
    base_plot_path = r'/nfs/masi/nathv/py_src_code_2020/dmipy_model_pictures'
    base_plot_path = os.path.normpath(base_plot_path)

    # Method Saving Paths
    base_save_path = r'/nfs/masi/nathv/miccai_2020/micro_methods_hcp_mini'
    base_save_path = os.path.normpath(base_save_path)

    # Create base saving path for Method
    # TODO The Method name can be made an argument later on
    method_name = 'AXCALIBER'

    # Base HCP Data Path
    base_data_path = r'/nfs/HCP/data'
    base_data_path = os.path.normpath(base_data_path)


    # Subject ID's list
    subj_ID_List = ['125525', '118225', '116726', '115825', '115017', '114823']
    # TODO When needed loop here over the ID list
    for subj_ID in subj_ID_List:
        # Subject Save Path
        subj_save_path = os.path.join(base_save_path, subj_ID)
        if os.path.exists(subj_save_path)==False:
            os.mkdir(subj_save_path)

        # TODO For later the subject data, bval and bvec reading part can be put inside a function
        subj_data_path = os.path.join(base_data_path, subj_ID_List[0], 'T1w', 'Diffusion')

        # Read the Nifti file, bvals and bvecs
        subj_bvals = np.loadtxt(os.path.join(subj_data_path, 'bvals'))
        subj_bvecs = np.loadtxt(os.path.join(subj_data_path, 'bvecs'))

        all_bvals = subj_bvals * 1e6
        all_bvecs = np.transpose(subj_bvecs)

        subj_Acq_Scheme = acquisition_scheme_from_bvalues(all_bvals, all_bvecs, delta=10.6*1e-3, Delta=43.1*1e-3, TE=89.5*1e-3)
        print(subj_Acq_Scheme.print_acquisition_info)

        print('Loading the Nifti Data ...')
        data_start_time = time.time()

        subj_babel_object = nib.load(os.path.join(subj_data_path, 'data.nii.gz'))
        subj_data = subj_babel_object.get_fdata()
        axial_slice_data = subj_data[58:60, 68:70, 60:62, :]

        mask_babel_object = nib.load(os.path.join(subj_data_path, 'nodif_brain_mask.nii.gz'))
        mask_data = mask_babel_object.get_fdata()
        axial_mask_slice_data = mask_data[58:60, 68:70, 60:62]

        data_end_time = time.time()
        data_time = np.int(np.round(data_end_time - data_start_time))

        print('Data Loaded ... Time Taken: {}'.format(data_end_time - data_start_time))
        print('The Data Dimensions are: {}'.format(subj_data.shape))

        #### AxCaliber Begin ####
        ball = gaussian_models.G1Ball()
        cylinder = cylinder_models.C4CylinderGaussianPhaseApproximation()
        gamma_cylinder = distribute_models.DD1GammaDistributed(models=[cylinder])

        axcaliber_gamma = modeling_framework.MultiCompartmentModel(models=[ball, gamma_cylinder])

        axcaliber_gamma.set_fixed_parameter('DD1GammaDistributed_1_C4CylinderGaussianPhaseApproximation_1_lambda_par', 1.7e-9)
        axcaliber_gamma.set_fixed_parameter('DD1GammaDistributed_1_C4CylinderGaussianPhaseApproximation_1_mu', [0, 0])

        print('Fitting the AxCaliber Model ...')
        fit_start_time = time.time()
        mcdmi_fit = axcaliber_gamma.fit(subj_Acq_Scheme,
                                        axial_slice_data,
                                        mask=axial_mask_slice_data,
                                        solver='mix',
                                        maxiter=100,
                                        use_parallel_processing=True,
                                        number_of_processors=32)

        fit_end_time = time.time()
        print('Model Fitting Completed ... Time Taken to fit: {}'.format(fit_end_time - fit_start_time))
        fit_time = np.int(np.round(fit_end_time - fit_start_time))

        fitted_parameters = mcdmi_fit.fitted_parameters

        # Get List of Estimated Parameter Names
        para_Names_list = []
        for key, value in fitted_parameters.items():
            para_Names_list.append(key)

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

        print('debug here')
    return None


if __name__ == '__main__':
    main()