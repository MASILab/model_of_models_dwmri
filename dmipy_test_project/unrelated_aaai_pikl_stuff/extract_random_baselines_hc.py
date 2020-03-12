import os
import numpy as np
import pickle

def main():

    base_path = r'/nfs/masi/nathv/nvidia_aaai_cvpr_pkls/aaai_exp_pkls/aaai_exp/hippocampus/results/pickle_files/svgd_d10_rep_5seed'
    base_path = os.path.normpath(base_path)

    pkl_f_list = os.listdir(base_path)

    # Active iteration for 20% and 30% are 21 and 38
    rand_nodel_act_iter_11_val = []
    rand_nodel_act_iter_3_val = []

    rand_nodel_act_iter_11_test = []
    rand_nodel_act_iter_3_test = []

    act_iter_21_name = 'model_21'
    act_iter_38_name = 'model_38'

    for each_pkl_f in pkl_f_list:
        pkl_f_path = os.path.join(base_path, each_pkl_f)

        # Read Pickle File
        infile = open(pkl_f_path, 'rb')
        data_dict = pickle.load(infile)

        rand_nodel_act_iter_3_val.append(data_dict['svgd_rand_nodel'][act_iter_21_name]['max_val_dice'])
        rand_nodel_act_iter_11_val.append(data_dict['svgd_rand_nodel'][act_iter_38_name]['max_val_dice'])

        rand_nodel_act_iter_3_test.append(data_dict['svgd_rand_nodel'][act_iter_21_name]['max_test_dice'])
        rand_nodel_act_iter_11_test.append(data_dict['svgd_rand_nodel'][act_iter_38_name]['max_test_dice'])

    print('At 20 % of the Data ... Validation')
    print(rand_nodel_act_iter_3_val)
    print(np.mean(rand_nodel_act_iter_3_val))
    print(np.std(rand_nodel_act_iter_3_val))

    print('At 20 % of the Data ... Test')
    print(rand_nodel_act_iter_3_test)
    print(np.mean(rand_nodel_act_iter_3_test))
    print(np.std(rand_nodel_act_iter_3_test))

    print('At 30 % of the Data ... Validation')
    print(rand_nodel_act_iter_11_val)
    print(np.mean(rand_nodel_act_iter_11_val))
    print(np.std(rand_nodel_act_iter_11_val))

    print('At 30 % of the Data ... Test')
    print(rand_nodel_act_iter_11_test)
    print(np.mean(rand_nodel_act_iter_11_test))
    print(np.std(rand_nodel_act_iter_11_test))

    print('Debug here')

    return None

if __name__=="__main__":
    main()