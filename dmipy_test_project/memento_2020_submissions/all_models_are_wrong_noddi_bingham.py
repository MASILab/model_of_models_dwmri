import os
import numpy as np
import time

# Dmipy imports
from dmipy.core.acquisition_scheme import acquisition_scheme_from_bvalues

# NODDI Watson Imports
from dmipy.signal_models import cylinder_models, gaussian_models
from dmipy.distributions.distribute_models import SD2BinghamDistributed
from dmipy.core.modeling_framework import MultiCompartmentModel

def main():

    # Base Path of all given files for All models are wrong
    base_path = r'/nfs/masi/nathv/memento_2020/all_models_are_wrong/files_project_2927_session_1436090/'
    base_path = os.path.normpath(base_path)

    # Just dealing with PGSE for now
    pgse_acq_params_path = os.path.join(base_path, 'PGSE_AcqParams.txt')
    pgse_signal_path = os.path.join(base_path, 'PGSE_Simulations.txt')

    # Read files via Numpy
    pgse_acq_params = np.loadtxt(pgse_acq_params_path)
    pgse_signal_data = np.loadtxt(pgse_signal_path)
    pgse_example_sub_diff = np.loadtxt('/nfs/masi/nathv/memento_2020/all_models_are_wrong/files_project_2927_session_1436090/2-AllModelsAreWrong-ExampleSubmissions/DIffusivity-ExampleSubmission3/PGSE.txt')
    pgse_example_sub_volfrac = np.loadtxt('/nfs/masi/nathv/memento_2020/all_models_are_wrong/files_project_2927_session_1436090/2-AllModelsAreWrong-ExampleSubmissions/VolumeFraction-ExampleSubmission3/PGSE.txt')

    # Transpose the Signal data
    pgse_signal_data = pgse_signal_data.transpose()

    # Dissect the acquisition parameters to form the Acquisition Table
    bvecs = pgse_acq_params[:, 1:4]
    bvals = pgse_acq_params[:, 6] * 1e6
    grad_str = pgse_acq_params[:, 0]
    small_del = pgse_acq_params[:, 4]
    big_del = pgse_acq_params[:, 5]

    subj_Acq_Scheme = acquisition_scheme_from_bvalues(bvals,
                                                      bvecs,
                                                      delta=small_del,
                                                      Delta=big_del
                                                      )

    print(subj_Acq_Scheme.print_acquisition_info)

    #### NODDI Watson ####
    ball = gaussian_models.G1Ball()
    stick = cylinder_models.C1Stick()
    zeppelin = gaussian_models.G2Zeppelin()

    bingham_dispersed_bundle = SD2BinghamDistributed(models=[stick, zeppelin])

    bingham_dispersed_bundle.set_tortuous_parameter('G2Zeppelin_1_lambda_perp', 'C1Stick_1_lambda_par',
                                                    'partial_volume_0')
    bingham_dispersed_bundle.set_equal_parameter('G2Zeppelin_1_lambda_par', 'C1Stick_1_lambda_par')
    bingham_dispersed_bundle.set_fixed_parameter('G2Zeppelin_1_lambda_par', 1.7e-9)

    NODDI_bingham_mod = MultiCompartmentModel(models=[ball, bingham_dispersed_bundle])
    NODDI_bingham_mod.set_fixed_parameter('G1Ball_1_lambda_iso', 3e-9)

    print('Fitting the NODDI Model ...')
    fit_start_time = time.time()
    NODDI_fit_hcp = NODDI_bingham_mod.fit(subj_Acq_Scheme,
                                          pgse_signal_data,
                                          use_parallel_processing=True,
                                          number_of_processors=8)
    
    fit_end_time = time.time()
    print('Model Fitting Completed ... Time Taken to fit: {}'.format(fit_end_time - fit_start_time))
    fit_time = np.int(np.round(fit_end_time - fit_start_time))

    sub_1_pv0 = NODDI_fit_hcp.fitted_parameters['partial_volume_0']
    sub_2_pv1 = NODDI_fit_hcp.fitted_parameters['partial_volume_1']

    np.savetxt('noddi_bingham_pv0.txt', sub_1_pv0)
    np.savetxt('noddi_bingham_pv1.txt', sub_2_pv1)

    print('Debug here')

    return None

if __name__=="__main__":
    main()