import os
import nibabel as nib
import numpy as np

def main():
    base_path = r'/nfs/masi/nathv/miccai_2020_hcp_100'
    base_path = os.path.normpath(base_path)

    plot_Save_Path = r'/nfs/masi/nathv/miccai_2020/microstructure_min_max'
    plot_Save_Path = os.path.normpath(plot_Save_Path)
    if os.path.exists(plot_Save_Path) == False:
        os.mkdir(plot_Save_Path)

    #methods_List = ['BS_2003', 'IVIM', 'MC_SMT', 'NODDI_WATSON', 'DTI']
    #subj_ID_List = ['125525', '118225', '116726', '115017', '114823']
    methods_list = ['BS_2003', 'MC_SMT']
    #subj_ID_List = ['100610', '102311', '102816', '104416', '105923', '108323', '109123', '111312', '111514', '115825']
    # Dictionary of Methods with Values containing the
    # name of the volume that needs to be normalized
    '''
    'BS_2003': ['C1Stick_1_mu.nii.gz',
                                  'C1Stick_1_lambda_par.nii.gz',
                                  'G1Ball_1_lambda_iso.nii.gz',
                                  'partial_volume_0.nii.gz',
                                  'partial_volume_1.nii.gz'],
                      'IVIM': ['partial_volume_1.nii.gz',
                               'G1Ball_1_lambda_iso.nii.gz', 'partial_volume_0.nii.gz'],
                      'MC_SMT': ['BundleModel_1_G2Zeppelin_1_lambda_par.nii.gz',
                                 'BundleModel_1_partial_volume_0.nii.gz'],
                      'NODDI_WATSON': ['SD1WatsonDistributed_1_SD1Watson_1_mu.nii.gz',
                                       'SD1WatsonDistributed_1_partial_volume_0.nii.gz',
                                       'SD1WatsonDistributed_1_SD1Watson_1_odi.nii.gz',
                                       'partial_volume_0.nii.gz',
                                       'partial_volume_1.nii.gz'],
    '''
    operating_Dict = {'BS_2003': ['C1Stick_1_mu.nii.gz',
                                  'C1Stick_1_lambda_par.nii.gz',
                                  'G1Ball_1_lambda_iso.nii.gz',
                                  'partial_volume_0.nii.gz',
                                  'partial_volume_1.nii.gz'],
                      'MC_SMT': ['BundleModel_1_G2Zeppelin_1_lambda_par.nii.gz',
                                 'BundleModel_1_partial_volume_0.nii.gz']
                      }

    subj_ID_List = os.listdir(base_path)

    # Loop over subjects
    for each_subject in subj_ID_List:

        print('Working on Subject: {} ...'.format(each_subject))

        # Load the Mask Data
        mask_path = os.path.join(base_path, each_subject, 'mask_data.nii.gz')
        mask_babel_obj = nib.load(mask_path)
        mask_data = mask_babel_obj.get_data()
        mask_data_boolean = mask_data.astype('bool')

        # Iterate over Methods & Metrics dictionary
        for method_key, metrics_list in operating_Dict.items():

            # This path is where the norm volumes will be saved
            saving_path = os.path.join(base_path, each_subject, method_key, 'Norm')
            if os.path.exists(saving_path) == False:
                os.mkdir(saving_path)

            # Looping over the metrics of a method
            for each_metric in metrics_list:

                # Metric Volume Path, the one to be loaded
                metric_vol_path = os.path.join(base_path, each_subject, method_key, each_metric)

                # Load the metric volume
                metric_vol_obj = nib.load(metric_vol_path)
                metric_vol = metric_vol_obj.get_fdata()

                # TODO If there are 2 metrics stored in a file then we segregate and normalize
                metric_dims = metric_vol.shape
                if len(metric_dims) == 3:

                    masked_metric = metric_vol[mask_data_boolean]

                    print('Min and Max Values before Norm ...')
                    print('Min value is {} and Max value is {}'.format(masked_metric.min(), masked_metric.max()))

                    # Perform Normalization
                    # Note that the minimum and maximum are being extracted through a masked array
                    # However, for simplicity we will use the estimated min and max on the entire unmasked volume
                    t_min = masked_metric.min()
                    t_max = masked_metric.max()

                    new_metric_vol = (metric_vol - t_min)/(t_max - t_min)
                    print('Min and Max Values after Norm ...')
                    print('Min value is {} and Max value is {}'.format(new_metric_vol.min(), new_metric_vol.max()))
                    print('#########')
                    # Save the new file by stealing Nifti header information from prior object
                    affine = metric_vol_obj.affine
                    new_img = nib.Nifti1Image(new_metric_vol, affine)

                    # Creating the new name for the normalized metric that will be estimated
                    new_metric_name = each_metric[:len(each_metric) - 7] + '_norm.nii.gz'

                    new_metric_save_path = os.path.join(saving_path, new_metric_name)
                    nib.save(new_img, new_metric_save_path)

                elif len(metric_dims) == 4:

                    # Segregate the Data
                    fourth_dim_len = metric_dims[3]

                    print('4th Dimensional Volumes detected, in total found: {}'.format(fourth_dim_len))

                    for each_part_metric in range(fourth_dim_len):

                        # We add 1 to avoid the zeroing factor as python is zero indexed
                        each_part_real_count = each_part_metric + 1

                        # Extracting the relevant metric
                        metric_inpart = metric_vol[:, :, :, each_part_metric]

                        masked_metric = metric_inpart[mask_data_boolean]

                        print('Min and Max Values before Norm ...')
                        print('Min value is {} and Max value is {}'.format(masked_metric.min(), masked_metric.max()))

                        # Perform Normalization
                        # Note that the minimum and maximum are being extracted through a masked array
                        # However, for simplicity we will use the estimated min and max on the entire unmasked volume
                        t_min = masked_metric.min()
                        t_max = masked_metric.max()

                        new_metric_vol = (metric_inpart - t_min) / (t_max - t_min)
                        print('Min and Max Values after Norm ...')
                        print('Min value is {} and Max value is {}'.format(new_metric_vol.min(), new_metric_vol.max()))
                        print('#########')
                        # Save the new file by stealing Nifti header information from prior object
                        affine = metric_vol_obj.affine
                        new_img = nib.Nifti1Image(new_metric_vol, affine)

                        # Assigning the new name to the metric
                        new_metric_name = each_metric[:len(each_metric) - 7] + '_part_{}'.format(each_part_real_count) + '_norm.nii.gz'

                        new_metric_save_path = os.path.join(saving_path, new_metric_name)
                        nib.save(new_img, new_metric_save_path)

    return None

if __name__=="__main__":
    main()