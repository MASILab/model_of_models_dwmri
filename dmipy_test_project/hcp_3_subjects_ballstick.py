import os
import numpy as np
import nibabel as nib
import time
import matplotlib.pyplot as plt

# Dmipy imports
from dmipy.core.acquisition_scheme import acquisition_scheme_from_bvalues

# Ball & Stick Imports
from dmipy.signal_models import cylinder_models, gaussian_models
from dmipy.core.modeling_framework import MultiCompartmentModel

# Ball & Racket Imports
from dmipy.signal_models import gaussian_models, cylinder_models
from dmipy.distributions.distribute_models import SD2BinghamDistributed
from dmipy.core.modeling_framework import MultiCompartmentModel

# IVIM Imports
from dmipy.signal_models.gaussian_models import G1Ball
from dmipy.core.modeling_framework import MultiCompartmentModel
from dmipy.custom_optimizers.intra_voxel_incoherent_motion import ivim_Dstar_fixed

# MSMT-CSD Imports
from dmipy.tissue_response.three_tissue_response import three_tissue_response_dhollander16

def main():

    # Plot Save Path
    base_plot_path = r'/nfs/masi/nathv/py_src_code_2020/dmipy_model_pictures'
    base_plot_path = os.path.normpath(base_plot_path)

    # Base HCP Data Path
    base_data_path = r'/nfs/HCP/data'
    base_data_path = os.path.normpath(base_data_path)

    # Subject ID's list
    subj_ID_List = ['125525', '118225', '116726']

    # TODO When needed loop here over the ID list
    subj_data_path = os.path.join(base_data_path, subj_ID_List[0], 'T1w', 'Diffusion')

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
    axial_slice_data = subj_data[:,:,65,:]

    data_end_time = time.time()
    data_time = np.int(np.round(data_end_time - data_start_time))

    print('Data Loaded ... Time Taken: {}'.format(data_end_time-data_start_time))
    print('The Data Dimensions are: {}'.format(subj_data.shape))

    # DMIPY Model Stuff
    '''
    #### Ball & Stick ####
    ball = gaussian_models.G1Ball()
    stick = cylinder_models.C1Stick()
    BAS_mod = MultiCompartmentModel(models=[stick, ball])

    print('Fitting the Ball & Stick Model ...')
    fit_start_time = time.time()
    BAS_fit_hcp = BAS_mod.fit(subj_Acq_Scheme, axial_slice_data, mask=axial_slice_data[..., 0] > 0)
    fit_end_time = time.time()
    print('Model Fitting Completed ... Time Taken to fit: {}'.format(fit_end_time - fit_start_time))
    fit_time = np.int(np.round(fit_end_time - fit_start_time))

    fitted_parameters = BAS_fit_hcp.fitted_parameters

    fig, axs = plt.subplots(2, 2, figsize=[10, 10])
    axs = axs.ravel()

    counter = 0
    for name, values in fitted_parameters.items():
        if values.squeeze().ndim != 2:
            continue
        cf = axs[counter].imshow(values.squeeze().T, origin=True, interpolation='nearest')
        axs[counter].set_title(name)
        axs[counter].set_axis_off()
        fig.colorbar(cf, ax=axs[counter], shrink=0.8)
        counter += 1

    bs_plt_name = 'ball_stick_behrens_{}.png'.format(fit_time)
    plt.savefig(os.path.join(base_plot_path, bs_plt_name))
    plt.clf()
    #### End of Ball & Stick ####
    '''

    '''
    #### Ball & Racket ####

    ball = gaussian_models.G1Ball()
    stick = cylinder_models.C1Stick()
    dispersed_stick = SD2BinghamDistributed([stick])
    BAR_mod = MultiCompartmentModel(models=[dispersed_stick, ball])

    # Parameter Fixing makes the model run faster
    BAR_mod.set_fixed_parameter("SD2BinghamDistributed_1_C1Stick_1_lambda_par", 1.7e-9)

    print('Fitting the Ball & Racket Model ...')
    fit_start_time = time.time()
    BAR_fit_hcp = BAR_mod.fit(subj_Acq_Scheme, axial_slice_data, mask=axial_slice_data[..., 0] > 0)
    fit_end_time = time.time()
    print('Model Fitting Completed ... Time Taken to fit: {}'.format(fit_end_time - fit_start_time))
    fit_time = np.int(np.round(fit_end_time - fit_start_time))

    fitted_parameters = BAR_fit_hcp.fitted_parameters

    fig, axs = plt.subplots(2, 3, figsize=[15, 10])
    axs = axs.ravel()

    counter = 0
    for name, values in fitted_parameters.items():
        if values.squeeze().ndim != 2:
            continue
        cf = axs[counter].imshow(values.squeeze().T, origin=True, interpolation='nearest')
        axs[counter].set_title(name)
        fig.colorbar(cf, ax=axs[counter], shrink=0.5)
        counter += 1

    br_plt_name = 'ball_racket_{}.png'.format(fit_time)
    plt.savefig(os.path.join(base_plot_path, br_plt_name))
    plt.clf()

    #### End of Ball & Racket Stuff ####
    '''

    '''
    #### IVIM ####
    print('Fitting IVIM ...')
    fit_start_time = time.time()
    ivim_fit_dmipy_fixed = ivim_Dstar_fixed(subj_Acq_Scheme, axial_slice_data, mask=axial_slice_data[..., 0] > 0)
    fit_end_time = time.time()
    print('IVIM Fitting Completed ... Time Taken to fit: {}'.format(fit_end_time - fit_start_time))
    fit_time = np.int(np.round(fit_end_time - fit_start_time))

    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=[10, 10])
    axs = axs.ravel()
    axs[0].set_title('Dmipy Dstar-Fixed', fontsize=18)
    axs[0].set_ylabel('S0-Predicted', fontsize=15)
    axs[1].set_ylabel('perfusion fraction', fontsize=15)
    axs[2].set_ylabel('D_star (perfusion)', fontsize=15)
    axs[3].set_ylabel('D (diffusion)', fontsize=15)

    args = {'vmin': 0., 'interpolation': 'nearest'}
    im0 = axs[0].imshow(ivim_fit_dmipy_fixed.S0, **args)
    im1 = axs[1].imshow(ivim_fit_dmipy_fixed.fitted_parameters['partial_volume_1'], vmax=1., **args)
    im2 = axs[2].imshow(np.ones_like(ivim_fit_dmipy_fixed.S0) *
                        ivim_fit_dmipy_fixed.fitted_and_linked_parameters['G1Ball_2_lambda_iso'] * 1e9, vmax=20, **args)
    axs[2].text(10, 10, 'Fixed to 7e-9 mm$^2$/s', fontsize=14, color='white')
    im3 = axs[3].imshow(ivim_fit_dmipy_fixed.fitted_parameters['G1Ball_1_lambda_iso'] * 1e9, vmax=6, **args)

    for im, ax in zip([im0, im1, im2, im3], axs):
        fig.colorbar(im, ax=ax, shrink=0.7)

    ivim_plt_name = 'ivim_{}.png'.format(fit_time)
    plt.savefig(os.path.join(base_plot_path, ivim_plt_name))
    plt.clf()

    #### End of IVIM ####
    '''

    ## TODO Investigate this bug, dimension error while fitting response functions
    #### MSMT-CSD ####
    S0_tissue_responses, tissue_response_models, selection_map = three_tissue_response_dhollander16(
        subj_Acq_Scheme, axial_slice_data, wm_algorithm='tournier13',
        wm_N_candidate_voxels=150, gm_perc=0.2, csf_perc=0.4)
    TR2_wm, TR1_gm, TR1_csf = tissue_response_models
    S0_wm, S0_gm, S0_csf = S0_tissue_responses

    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=[10, 10])

    axs = axs.ravel()
    axs[0].set_title('Dmipy Dstar-Fixed', fontsize=18)

    args = {'vmin': 0., 'interpolation': 'nearest'}
    im0 = axs[0].imshow(axial_slice_data[:,:,0,0], origin=True)
    im0 = axs[0].imshow(selection_map.squeeze(), origin=True, alpha=0.8)

    plt.show()

    #### End of MSMT-CSD ####

    #### MC MDI CSD ####
    
    print('Debug here')
    return None

if __name__ == '__main__':
    main()