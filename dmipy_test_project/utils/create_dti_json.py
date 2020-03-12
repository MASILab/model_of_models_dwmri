import os
import json

def main():

    base_path = r'/nfs/masi/nathv/miccai_2020/micro_methods_hcp_mini'
    base_path = os.path.normpath(base_path)

    methods_List = ['DTI']
    subj_ID_List = ['100610', '102311', '102816',
                    '104416', '105923', '108323',
                    '109123', '111312', '111514',
                    '125525', '118225', '116726',
                    '114823', '115017', '115825']

    # Create a Method-Metric Dictionary
    t_s_path = os.path.join(base_path, '125525')
    method_metric_dict = {}

    ### Loop over methods
    for each_m in methods_List:

        # Get a list of all file names
        metric_f_names = os.listdir(os.path.join(t_s_path, each_m, 'Norm'))

        # Store as key value in method metric dict
        method_metric_dict[each_m] = metric_f_names

    json_f_name = r'data_list_dti.json'
    json_path = os.path.join(base_path, json_f_name)

    # Filenames are hard-coded
    sh_f_name = r'DWI1K_sh_masked.nii'
    mask_name = r'nodif_brain_mask.nii'

    json_dump = []

    for each_subject in subj_ID_List:

        subj_path = os.path.join(base_path, each_subject)

        # Input Image
        sh_f_path = os.path.join(subj_path, sh_f_name)

        # Mask Image
        mask_path = os.path.join(subj_path, mask_name)

        output_dict = {}

        # Constructing output dictionary
        for key, value in method_metric_dict.items():

            metric_path_list = []
            # Loop over metrics
            for each_metric in value:
                metric_path = os.path.join(subj_path, key, 'Norm', each_metric)
                metric_path_list.append(metric_path)

            output_dict[key] = metric_path_list

        data_dict = {'input_image': sh_f_path, 'mask': mask_path, 'output': output_dict}
        json_dump.append(data_dict)

    new_data_dict = {}
    new_data_dict['train'] = json_dump[0:9]
    new_data_dict['validation'] = json_dump[9:14]
    new_data_dict['test'] = json_dump[14]

    with open(json_path, 'w') as json_file:
        json.dump(new_data_dict, json_file)
    json_file.close()

    return None

if __name__=="__main__":
    main()

