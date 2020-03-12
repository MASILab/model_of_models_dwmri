import os
import numpy as np
import pickle
import matplotlib.pyplot as plt

def main():
    base_path = r'/nfs/masi/nathv/miccai_2020/bottleneck_compart_dti'
    base_path = os.path.normpath(base_path)

    # TODO Extract all key loss names from a pickle file
    overall_dict = {}
    temp_pkl_name = r'/nfs/masi/nathv/miccai_2020/bottleneck_compart_dti/bn_1/bn_1_metric.pkl'
    t_pkl_data = pickle.load(open(temp_pkl_name, "rb"))
    # Iterate over to create a dictionary with empty lists for all keys
    for key, value in t_pkl_data.items():
        overall_dict[key] = []

    # TODO Even though the below line exists, the directories were not read in an ascending order
    base_dir_list = os.listdir(base_path)

    # Loop for first 15 bottleneck
    # Python range is exclusive of the last number
    for bn_number in range(1, 16):
        # Construct the bottleneck name path
        bn_dir_name = 'bn_{}'.format(bn_number)

        # Construct the pickle metric file name
        bn_pickle_name = 'bn_{}_metric.pkl'.format(bn_number)

        # The Pickled Path
        pickle_f_path = os.path.join(base_path, bn_dir_name, bn_pickle_name)

        # Loading the pickle
        pickle_data = pickle.load(open(pickle_f_path, "rb"))

        # Iterate over bottleneck's pickle data and store the min value per loss value list
        for pickle_key, pickle_list in pickle_data.items():

            # Grab the list
            t_list = pickle_data[pickle_key]
            overall_dict[pickle_key].append(np.min(t_list))

    print('Accumulation Stuff done, now we plot')

    plt.figure(1, figsize=(14, 7))
    plt.subplot(2, 3, 1)
    plt.plot(overall_dict['loss'])
    plt.plot(overall_dict['val_loss'])
    plt.xlabel('Bottleneck Number')
    plt.ylabel('Mean Squared Error')
    plt.grid()
    plt.legend(['Training', 'Validation'])
    plt.title('Overall Loss')

    plt.subplot(2, 3, 2)
    plt.plot(overall_dict['dti_loss'])
    plt.plot(overall_dict['val_dti_loss'])
    plt.xlabel('Bottleneck Number')
    plt.ylabel('Mean Squared Error')
    plt.grid()
    plt.legend(['Training', 'Validation'])
    plt.title('DTI Loss')

    plt.subplot(2, 3, 3)
    plt.plot(overall_dict['ivim_loss'])
    plt.plot(overall_dict['val_ivim_loss'])
    plt.xlabel('Bottleneck Number')
    plt.ylabel('Mean Squared Error')
    plt.grid()
    plt.legend(['Training', 'Validation'])
    plt.title('IVIM Loss')

    plt.subplot(2, 3, 4)
    plt.plot(overall_dict['mc_smt_loss'])
    plt.plot(overall_dict['val_mc_smt_loss'])
    plt.xlabel('Bottleneck Number')
    plt.ylabel('Mean Squared Error')
    plt.grid()
    plt.legend(['Training', 'Validation'])
    plt.title('MC-SMT Loss')

    plt.subplot(2, 3, 5)
    plt.plot(overall_dict['ball_stick_loss'])
    plt.plot(overall_dict['val_ball_stick_loss'])
    plt.xlabel('Bottleneck Number')
    plt.ylabel('Mean Squared Error')
    plt.grid()
    plt.legend(['Training', 'Validation'])
    plt.title('Ball Stick Loss')

    plt.subplot(2, 3, 6)
    plt.plot(overall_dict['noddi_loss'])
    plt.plot(overall_dict['val_noddi_loss'])
    plt.xlabel('Bottleneck Number')
    plt.ylabel('Mean Squared Error')
    plt.grid()
    plt.legend(['Training', 'Validation'])
    plt.title('NODDI Loss')
    plt.clf()
    plt.close(1)

    plt.figure(2, figsize=(8,4))
    plt.subplot(1,2,1)
    plt.plot(overall_dict['dti_loss'])
    plt.plot(overall_dict['ivim_loss'])
    plt.plot(overall_dict['mc_smt_loss'])
    plt.plot(overall_dict['ball_stick_loss'])
    plt.plot(overall_dict['noddi_loss'])
    plt.ylim([0.0, 0.05])
    plt.grid()
    plt.xlabel('Bottleneck Number')
    plt.ylabel('Mean Squared Error')
    plt.legend(['DTI', 'IVIM', 'MC-SMT', 'Ball-Stick', 'NODDI'])
    plt.title('All Method Training Losses')


    plt.subplot(1, 2, 2)
    plt.plot(overall_dict['val_dti_loss'], '--')
    plt.plot(overall_dict['val_ivim_loss'], '--')
    plt.plot(overall_dict['val_mc_smt_loss'], '--')
    plt.plot(overall_dict['val_ball_stick_loss'], '--')
    plt.plot(overall_dict['val_noddi_loss'], '--')
    plt.ylim([0.0, 0.05])
    plt.xlabel('Bottleneck Number')
    plt.ylabel('Mean Squared Error')
    plt.grid()
    plt.legend(['DTI', 'IVIM', 'MC-SMT', 'Ball-Stick', 'NODDI'])
    plt.title('All Method Validation Losses')

    plt.show()
    print('Debug here')

    return None

if __name__=="__main__":
    main()