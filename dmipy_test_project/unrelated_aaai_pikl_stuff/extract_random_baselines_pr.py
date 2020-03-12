import os
import numpy as np
import pickle

def main():

    base_path = r'/nfs/masi/nathv/nvidia_aaai_cvpr_pkls/aaai_exp_pkls/aaai_exp/pancreas/results/pickle_files/svgd_d20_rep_5seed'
    base_path = os.path.normpath(base_path)

    pkl_f_list = os.listdir(base_path)

    # Active iteration for 20% & 40% & 50% are 3 and 11 and 15
    rand_nodel_act_iter_15_val = []
    rand_nodel_act_iter_11_val = []
    rand_nodel_act_iter_3_val = []

    rand_nodel_act_iter_15_test = []
    rand_nodel_act_iter_11_test = []
    rand_nodel_act_iter_3_test = []

    act_iter_3_name = 'model_3'
    act_iter_11_name = 'model_11'
    act_iter_15_name = 'model_15'

    for each_pkl_f in pkl_f_list:
        pkl_f_path = os.path.join(base_path, each_pkl_f)

        # Read Pickle File
        infile = open(pkl_f_path, 'rb')
        data_dict = pickle.load(infile)

        rand_nodel_act_iter_3_val.append(data_dict['svgd_rand_nodel'][act_iter_3_name]['max_val_dice'])
        rand_nodel_act_iter_11_val.append(data_dict['svgd_rand_nodel'][act_iter_11_name]['max_val_dice'])
        rand_nodel_act_iter_15_val.append(data_dict['svgd_rand_nodel'][act_iter_15_name]['max_val_dice'])

        rand_nodel_act_iter_3_test.append(data_dict['svgd_rand_nodel'][act_iter_3_name]['max_val_test_dice'])
        rand_nodel_act_iter_11_test.append(data_dict['svgd_rand_nodel'][act_iter_11_name]['max_val_test_dice'])
        rand_nodel_act_iter_15_test.append(data_dict['svgd_rand_nodel'][act_iter_15_name]['max_val_test_dice'])

    print('At 20 % of the Data ... Validation')
    print(rand_nodel_act_iter_3_val)
    print(np.mean(rand_nodel_act_iter_3_val))
    print(np.std(rand_nodel_act_iter_3_val))

    print('At 20 % of the Data ... Test')
    print(rand_nodel_act_iter_3_test)
    print(np.mean(rand_nodel_act_iter_3_test))
    print(np.std(rand_nodel_act_iter_3_test))

    print('At 40 % of the Data ... Validation')
    print(rand_nodel_act_iter_11_val)
    print(np.mean(rand_nodel_act_iter_11_val))
    print(np.std(rand_nodel_act_iter_11_val))

    print('At 40 % of the Data ... Test')
    print(rand_nodel_act_iter_11_test)
    print(np.mean(rand_nodel_act_iter_11_test))
    print(np.std(rand_nodel_act_iter_11_test))

    print('At 50 % of the Data ... Validation')
    print(rand_nodel_act_iter_15_val)
    print(np.mean(rand_nodel_act_iter_15_val))
    print(np.std(rand_nodel_act_iter_15_val))

    print('At 50 % of the Data ... Test')
    print(rand_nodel_act_iter_15_test)
    print(np.mean(rand_nodel_act_iter_15_test))
    print(np.std(rand_nodel_act_iter_15_test))

    print('Debug here')
    return None

if __name__=="__main__":
    main()