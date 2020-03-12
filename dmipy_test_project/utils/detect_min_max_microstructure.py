import os
import json
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

def main():

    base_path = r'/nfs/masi/nathv/miccai_2020/micro_methods_hcp_mini'
    base_path = os.path.normpath(base_path)

    plot_Save_Path = r'/nfs/masi/nathv/miccai_2020/microstructure_min_max'
    plot_Save_Path = os.path.normpath(plot_Save_Path)
    if os.path.exists(plot_Save_Path) == False:
        os.mkdir(plot_Save_Path)

    methods_List = ['BS_2003', 'IVIM', 'MC_SMT', 'NODDI_WATSON']
    subj_ID_List = ['125525', '118225', '116726']

    #Iterate over Methods
    for each_method in methods_List:

        print('Working on the Method: {}'.format(each_method))
        # Get a list of the metrics
        list_Metrics = os.listdir(os.path.join(base_path, '125525', each_method))

        # Iterate over metrics
        for each_metric in list_Metrics:

            print('We are going to observe the metrics of : {}'.format(each_metric))

            # Iterate over subjects
            for each_subject in subj_ID_List:
                print('Working on subject ID: {}'.format(each_subject))
                subj_Metric_Path = os.path.join(base_path, each_subject, each_method, each_metric)

                # Read the metric file using Nibabel
                metric_vol_nifti_object = nib.load(subj_Metric_Path)
                metric_vol_data = metric_vol_nifti_object.get_fdata()

                print('The minimum value is {} and the maximum value is {}'.format(metric_vol_data.min(), metric_vol_data.max()))

            print('End of Metric #######')

        print('End of Method #######')






    return None

if __name__=="__main__":
    main()