import os
import numpy as np

from scipy.optimize import minimize
from scipy.linalg import pinv

from dipy.reconst.shore import ShoreModel
from dipy.reconst.shore import shore_matrix
from dipy.core.gradients import gradient_table

import matplotlib.pyplot as plt


def main():
    # Base Path of all the given files for DDE Part
    base_path = r'/nfs/masi/nathv/memento_2020/signal_forecast_data/files_project_2927_session_1436088/1-SignalForecast-ProvidedData/DODE'
    base_path = os.path.normpath(base_path)

    save_path = r'/nfs/masi/nathv/memento_2020/dode_submissions'
    save_path = os.path.normpath(save_path)
    if os.path.exists(save_path) == False:
        os.mkdir(save_path)

    # Read files via numpy load txt
    acq_params_file = np.loadtxt(os.path.join(base_path, 'DODE_provided_acq_params.txt'))
    pgse_provided_signals = np.loadtxt(os.path.join(base_path, 'DODE_provided_signals.txt'))
    pgse_unprovided_signals = np.loadtxt(os.path.join(base_path, 'DODE_unprovided_acq_params.txt'))

    print('All Relevant Files Read ...')

    # Extract the acquisition parameters from provided and
    # unprovided to form the two different basis sets for SHORE
    prov_bvecs_1 = acq_params_file[:, 1:4]
    prov_bvecs_2 = acq_params_file[:, 4:7]
    prov_bvals = acq_params_file[:, 12]

    unprov_bvecs_1 = pgse_unprovided_signals[:, 1:4]
    unprov_bvecs_2 = pgse_unprovided_signals[:, 4:7]
    unprov_bvals = pgse_unprovided_signals[:, 12]

    # Transposing the provided signals for fitting to SHORE
    prov_signals = pgse_provided_signals.transpose()

    prov_gtab_1 = gradient_table(prov_bvals, prov_bvecs_1)
    prov_gtab_2 = gradient_table(prov_bvals, prov_bvecs_2)

    unprov_gtab_1 = gradient_table(unprov_bvals, unprov_bvecs_1)
    unprov_gtab_2 = gradient_table(unprov_bvals, unprov_bvecs_2)

    print('Gradient Tables formed ...')
    # Default SHORE parameters are
    zeta = 700
    lambda_n = 1e-8
    lambda_l = 1e-8
    radial_order = 6

    # SHORE Regularization Matrix Initialization

    def l_shore(radial_order):
        "Returns the angular regularisation matrix for SHORE basis"
        F = radial_order / 2
        n_c = int(np.round(1 / 6.0 * (F + 1) * (F + 2) * (4 * F + 3)))
        diagL = np.zeros(n_c)
        counter = 0
        for l in range(0, radial_order + 1, 2):
            for n in range(l, int((radial_order + l) / 2) + 1):
                for m in range(-l, l + 1):
                    diagL[counter] = (l * (l + 1)) ** 2
                    counter += 1

        return np.diag(diagL)

    def n_shore(radial_order):
        "Returns the angular regularisation matrix for SHORE basis"
        F = radial_order / 2
        n_c = int(np.round(1 / 6.0 * (F + 1) * (F + 2) * (4 * F + 3)))
        diagN = np.zeros(n_c)
        counter = 0
        for l in range(0, radial_order + 1, 2):
            for n in range(l, int((radial_order + l) / 2) + 1):
                for m in range(-l, l + 1):
                    diagN[counter] = (n * (n + 1)) ** 2
                    counter += 1

        return np.diag(diagN)

    print('Minimizing the zeta scale Parameter for Input Data ...')

    def eval_shore(D, n, scale, input_gtab):
        lambdaN = 1e-8
        lambdaL = 1e-8
        radial_order = n

        Lshore = l_shore(radial_order)
        Nshore = n_shore(radial_order)

        M_1 = shore_matrix(n, scale, input_gtab, 1. / (4 * np.pi ** 2))
        #M_2 = shore_matrix(n, scale, prov_gtab_1, 1. / (4 * np.pi ** 2))

        MpseudoInv_1 = np.dot(np.linalg.inv(np.dot(M_1.T, M_1) + lambdaN * Nshore + lambdaL * Lshore), M_1.T)
        #MpseudoInv_2 = np.dot(np.linalg.inv(np.dot(M_2.T, M_2) + lambdaN * Nshore + lambdaL * Lshore), M_2.T)

        shorecoefs_1 = np.dot(D, MpseudoInv_1.T)
        #shorecoefs_2 = np.dot(D, MpseudoInv_2.T)

        # shorecoefs = np.dot(D, pinv(Mshore).T)
        shorepred_1 = np.dot(shorecoefs_1, M_1.T)
        #shorepred_2 = np.dot(shorecoefs_2, M_2.T)

        '''
        plt.figure(1, figsize=(12,8))
        plt.subplot(2,3,1)
        plt.plot(D[0, :])
        plt.plot(shorepred[0, :])
        plt.legend(['Original Signal','SHORE Fitted Signal'])

        plt.subplot(2,3,2)
        plt.plot(D[1, :])
        plt.plot(shorepred[1, :])

        plt.subplot(2,3,3)
        plt.plot(D[2, :])
        plt.plot(shorepred[2, :])

        plt.subplot(2,3,4)
        plt.plot(D[3, :])
        plt.plot(shorepred[3, :])

        plt.subplot(2,3,5)
        plt.plot(D[4, :])
        plt.plot(shorepred[4, :])
        plt.show()
        '''
        return np.linalg.norm(D - shorepred_1) ** 2

    def get_shore_coeffs(D, n, scale, input_gtab):
        lambdaN = 1e-8
        lambdaL = 1e-8
        radial_order = n

        Lshore = l_shore(radial_order)
        Nshore = n_shore(radial_order)

        M = shore_matrix(n, scale, input_gtab, 1. / (4 * np.pi ** 2))
        MpseudoInv = np.dot(np.linalg.inv(np.dot(M.T, M) + lambdaN * Nshore + lambdaL * Lshore), M.T)
        shorecoefs = np.dot(D, MpseudoInv.T)
        return shorecoefs

    zeta_optimized_1 = minimize(lambda x: eval_shore(prov_signals, radial_order, x, prov_gtab_1), zeta)['x']
    zeta_optimized_2 = minimize(lambda x: eval_shore(prov_signals, radial_order, x, prov_gtab_2), zeta)['x']
    print('The guess zeta value is: {} and the optimal zeta values are: {} & {}'.format(zeta, zeta_optimized_1, zeta_optimized_2))

    guess_prov_mse_1 = eval_shore(prov_signals, radial_order, zeta, prov_gtab_1)
    guess_prov_mse_2 = eval_shore(prov_signals, radial_order, zeta, prov_gtab_2)

    optim_prov_mse_1 = eval_shore(prov_signals, radial_order, zeta_optimized_1, prov_gtab_1)
    optim_prov_mse_2 = eval_shore(prov_signals, radial_order, zeta_optimized_2, prov_gtab_2)

    guess_shore_coeffs_1 = get_shore_coeffs(prov_signals, radial_order, zeta, prov_gtab_1)
    guess_shore_coeffs_2 = get_shore_coeffs(prov_signals, radial_order, zeta, prov_gtab_2)

    optim_shore_coeffs_1 = get_shore_coeffs(prov_signals, radial_order, zeta_optimized_1, prov_gtab_1)
    optim_shore_coeffs_2 = get_shore_coeffs(prov_signals, radial_order, zeta_optimized_2, prov_gtab_2)

    print('MSE for guessed zeta is: {} & {} and MSE for optimized zeta is: {} & {}'.format(guess_prov_mse_1, guess_prov_mse_2, optim_prov_mse_1, optim_prov_mse_2))

    print('Obtaining signal measurements for unprovided data')
    M_unprov_zeta_1 = shore_matrix(radial_order, zeta, unprov_gtab_1, 1. / (4 * np.pi ** 2))
    M_unprov_zeta_2 = shore_matrix(radial_order, zeta, unprov_gtab_2, 1. / (4 * np.pi ** 2))

    M_unprov_zeta_optim_1 = shore_matrix(radial_order, zeta_optimized_1, unprov_gtab_1, 1. / (4 * np.pi ** 2))
    M_unprov_zeta_optim_2 = shore_matrix(radial_order, zeta_optimized_2, unprov_gtab_2, 1. / (4 * np.pi ** 2))

    guess_shore_preds_1 = np.dot(guess_shore_coeffs_1, M_unprov_zeta_1.T)
    guess_shore_preds_2 = np.dot(guess_shore_coeffs_2, M_unprov_zeta_2.T)

    optim_shore_preds_1 = np.dot(optim_shore_coeffs_1, M_unprov_zeta_optim_1.T)
    optim_shore_preds_2 = np.dot(optim_shore_coeffs_2, M_unprov_zeta_optim_2.T)

    guess_shore_preds = (guess_shore_preds_1.T + guess_shore_preds_2.T)/2.0
    optim_shore_preds = (optim_shore_preds_1.T + optim_shore_preds_2.T)/2.0


    np.savetxt(os.path.join(save_path, 'sub_1.txt'), guess_shore_preds)
    np.savetxt(os.path.join(save_path, 'sub_2.txt'), optim_shore_preds)
    '''
    prov_ShoreModel = ShoreModel(gtab=prov_gtab,
                                 radial_order=radial_order,
                                 zeta=zeta,
                                 lambdaN=lambda_n,
                                 lambdaL=lambda_l)

    print('Shore Basis Constructed, Fitting Data ...')
    prov_ShoreFit = prov_ShoreModel.fit(pgse_provided_signals.transpose())
    '''

    print('Debug here')

    return None


if __name__ == "__main__":
    main()