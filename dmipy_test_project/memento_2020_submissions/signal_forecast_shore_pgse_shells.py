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
    base_path = r'/nfs/masi/nathv/memento_2020/signal_forecast_data/files_project_2927_session_1436088/1-SignalForecast-ProvidedData/PGSE_shells'
    base_path = os.path.normpath(base_path)

    save_path = r'/nfs/masi/nathv/memento_2020/pgse_shells_submissions'
    save_path = os.path.normpath(save_path)

    # Read files via numpy load txt
    acq_params_file = np.loadtxt(os.path.join(base_path, 'PGSE_shells_provided_acq_params.txt'))
    pgse_provided_signals = np.loadtxt(os.path.join(base_path, 'PGSE_shells_provided_signals.txt'))
    pgse_unprovided_signals = np.loadtxt(os.path.join(base_path, 'PGSE_shells_unprovided_acq_params.txt'))

    print('All Relevant Files Read ...')

    # Extract the acquisition parameters from provided and
    # unprovided to form the two different basis sets for SHORE
    prov_bvecs = acq_params_file[:, 1:4]
    prov_bvals = acq_params_file[:, 9]

    unprov_bvecs = pgse_unprovided_signals[:, 1:4]
    unprov_bvals = pgse_unprovided_signals[:, 9]

    # Transposing the provided signals for fitting to SHORE
    prov_signals = pgse_provided_signals.transpose()

    prov_gtab = gradient_table(prov_bvals, prov_bvecs)
    unprov_gtab = gradient_table(unprov_bvals, unprov_bvecs)

    print('Gradient Tables formed ...')
    # Default SHORE parameters are
    zeta = 700
    lambda_n = 1e-8
    lambda_l = 1e-8
    radial_order = 18

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

    def eval_shore(D, n, scale):
        lambdaN = 1e-8
        lambdaL = 1e-8
        radial_order = n

        Lshore = l_shore(radial_order)
        Nshore = n_shore(radial_order)

        M = shore_matrix(n, scale, prov_gtab, 1. / (4 * np.pi ** 2))
        MpseudoInv = np.dot(np.linalg.inv(np.dot(M.T, M) + lambdaN * Nshore + lambdaL * Lshore), M.T)
        shorecoefs = np.dot(D, MpseudoInv.T)

        # shorecoefs = np.dot(D, pinv(Mshore).T)
        shorepred = np.dot(shorecoefs, M.T)

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
        return np.linalg.norm(D - shorepred) ** 2

    def get_shore_coeffs(D, n, scale):
        lambdaN = 1e-8
        lambdaL = 1e-8
        radial_order = n

        Lshore = l_shore(radial_order)
        Nshore = n_shore(radial_order)

        M = shore_matrix(n, scale, prov_gtab, 1. / (4 * np.pi ** 2))
        MpseudoInv = np.dot(np.linalg.inv(np.dot(M.T, M) + lambdaN * Nshore + lambdaL * Lshore), M.T)
        shorecoefs = np.dot(D, MpseudoInv.T)
        return shorecoefs

    zeta_optimized = minimize(lambda x: eval_shore(prov_signals, radial_order, x), zeta)['x']
    print('The guess zeta value is: {} and the optimal zeta value is: {}'.format(zeta, zeta_optimized))

    guess_prov_mse = eval_shore(prov_signals, radial_order, zeta)
    optim_prov_mse = eval_shore(prov_signals, radial_order, zeta_optimized)

    guess_shore_coeffs = get_shore_coeffs(prov_signals, radial_order, zeta)
    optim_shore_coeffs = get_shore_coeffs(prov_signals, radial_order, zeta_optimized)

    print('MSE for guessed zeta is: {} and MSE for optimized zeta is: {}'.format(guess_prov_mse, optim_prov_mse))

    print('Obtaining signal measurements for unprovided data')
    M_unprov_zeta = shore_matrix(radial_order, zeta, unprov_gtab, 1. / (4 * np.pi ** 2))
    M_unprov_zeta_optim = shore_matrix(radial_order, zeta_optimized, unprov_gtab, 1. / (4 * np.pi ** 2))

    guess_shore_preds = np.dot(guess_shore_coeffs, M_unprov_zeta.T)
    optim_shore_preds = np.dot(optim_shore_coeffs, M_unprov_zeta_optim.T)

    guess_shore_preds = guess_shore_preds.T
    optim_shore_preds = optim_shore_preds.T

    #np.savetxt(os.path.join(save_path,'sub_1.txt'), guess_shore_preds)
    np.savetxt(os.path.join(save_path, 'sub_20th_order.txt'), optim_shore_preds)
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

if __name__=="__main__":
    main()