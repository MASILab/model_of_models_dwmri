import numpy as np
import nibabel as nib
import random
import time

def load_nifty(path_to_file, data_type):
    start_time = time.time()
    nifti_data = nib.load(path_to_file)
    nifti_img = nifti_data.get_fdata(dtype=data_type)
    #nifti_data.uncache()
    end_time = time.time()
    print('\n Time Take to Read {}'.format(end_time - start_time))
    return nifti_img

def nifti_image_generator_patch(inputPath, bs, patch_size):
    # open the CSV file for reading
    #f = open(inputPath, "r")
    n_retrievals = 25000
    n_classes = 45
    # loop indefinitely
    while True:

        for vol_index, each_vol in enumerate(inputPath):

            # Load Nifti Volumes of Input, Output and Mask
            input_vol = load_nifty(each_vol['input_image'], data_type='float32')
            mask_vol = load_nifty(each_vol['mask'], data_type='float32')
            dims = mask_vol.shape
            # Convert mask_vol to int to save space
            mask_vol = np.int16(mask_vol)

            # TODO Output volumes since there are many, this needs to be handled a bit carefully
            # TODO The output object is a dictionary and contains 4 models by keys of
            # TODO 'BS_2003', 'IVIM', 'MC_SMT', 'NODDI_WATSON' (The Ordering is important)
            output_object = each_vol['output']
            ms_methods_list = ['BS_2003', 'IVIM', 'MC_SMT', 'NODDI_WATSON']

            # Accumulation of all metrics for all methods in a dictionary by name
            accum_all_output = {}

            for each_method in ms_methods_list:

                print('Reading Files of Method: {}'.format(each_method))

                # Grab list of ms metrics for the method
                metric_list = output_object[each_method]
                num_metrics = len(metric_list)

                # Create an accumulated empty volume for metrics
                method_accum_vol = np.zeros((dims[0], dims[1], dims[2], num_metrics))

                # Accumulate by looping over the metrics
                for metric_index, each_metric in enumerate(metric_list):
                    print('File name being loaded ... {}'.format(each_metric))
                    t_vol = load_nifty(each_metric, data_type='float32')

                    if len(t_vol.shape) == 3:
                        method_accum_vol[:,:,:,metric_index] = t_vol
                    elif len(t_vol.shape) == 4:
                        method_accum_vol[:, :, :, metric_index] = t_vol[:,:,:,0
                                                                  ]
                # Accumulate all the scalar metric volumes one by one
                accum_all_output[each_method] = method_accum_vol

            # Extract Voxel Indices
            true_vox_inds = np.where(mask_vol == 1)
            true_vox_inds = np.asarray(true_vox_inds)
            true_vox_inds = np.transpose(true_vox_inds)
            len_true_vox = len(true_vox_inds)

            current_retrieval = 0
            while current_retrieval < n_retrievals:
                # initialize our batches of images and labels
                images = np.empty((bs, patch_size[0], patch_size[1], patch_size[2], n_classes))

                labels_bs = np.empty((bs, 5))
                labels_ivim = np.empty((bs, 3))
                labels_mcsmt = np.empty((bs, 2))
                labels_noddi = np.empty((bs, 5))

                # X = np.empty((self.batch_size, self.n_classes))
                # y = np.empty((self.batch_size, self.n_classes))

                # Generate Random Inds for usage
                rand_inds = random.sample(range(len_true_vox - 1), bs)

                # Generate data
                for each_ind, each in enumerate(rand_inds):
                    # Retrieve indices
                    vox_inds = true_vox_inds[each, :]
                    images[each_ind, :] = np.squeeze(input_vol[vox_inds[0]-1:vox_inds[0]+2,
                                                               vox_inds[1]-1:vox_inds[1]+2,
                                                               vox_inds[2]-1:vox_inds[2]+2, :])

                    labels_bs[each_ind, :] = np.squeeze(
                        accum_all_output['BS_2003'][vox_inds[0], vox_inds[1], vox_inds[2], :])
                    labels_ivim[each_ind, :] = np.squeeze(
                        accum_all_output['IVIM'][vox_inds[0], vox_inds[1], vox_inds[2], :])
                    labels_mcsmt[each_ind, :] = np.squeeze(
                        accum_all_output['MC_SMT'][vox_inds[0], vox_inds[1], vox_inds[2], :])
                    labels_noddi[each_ind, :] = np.squeeze(
                        accum_all_output['NODDI_WATSON'][vox_inds[0], vox_inds[1], vox_inds[2], :])

                current_retrieval = current_retrieval + bs
                yield (images, {'ball_stick': labels_bs, 'ivim': labels_ivim, 'mc_smt':labels_mcsmt, 'noddi':labels_noddi})


def nifti_compart_orient_generator(inputPath, bs, patch_size):
    # open the CSV file for reading
    #f = open(inputPath, "r")
    n_retrievals = 250000
    n_classes = 45
    # loop indefinitely
    while True:

        for vol_index, each_vol in enumerate(inputPath):

            # Load Nifti Volumes of Input, Output and Mask
            input_vol = load_nifty(each_vol['input_image'], data_type='float32')
            mask_vol = load_nifty(each_vol['mask'], data_type='float32')
            dims = mask_vol.shape
            # Convert mask_vol to int to save space
            mask_vol = np.int16(mask_vol)

            # TODO Output volumes since there are many, this needs to be handled a bit carefully
            # TODO The output object is a dictionary and contains 4 models by keys of
            # TODO 'BS_2003', 'IVIM', 'MC_SMT', 'NODDI_WATSON' (The Ordering is important)
            output_object = each_vol['output']
            ms_methods_list = ['BS_2003', 'IVIM', 'MC_SMT', 'NODDI_WATSON', 'DTI', 'MT_CSD']

            # Accumulation of all metrics for all methods in a dictionary by name
            accum_all_output = {}

            for each_method in ms_methods_list:

                print('Reading Files of Method: {}'.format(each_method))

                if each_method == 'MT_CSD':
                    # Grab list of ms metrics for the method
                    metric_list = output_object[each_method]
                    num_metrics = len(metric_list)

                    method_accum_vol = np.zeros((dims[0], dims[1], dims[2], 45))

                else:
                    # Grab list of ms metrics for the method
                    metric_list = output_object[each_method]
                    num_metrics = len(metric_list)

                    # Create an accumulated empty volume for metrics
                    method_accum_vol = np.zeros((dims[0], dims[1], dims[2], num_metrics))

                    # Accumulate by looping over the metrics
                    for metric_index, each_metric in enumerate(metric_list):
                        print('File name being loaded ... {}'.format(each_metric))
                        t_vol = load_nifty(each_metric, data_type='float32')

                        if len(t_vol.shape) == 3:
                            method_accum_vol[:, :, :, metric_index] = t_vol
                        else:
                            print('Found a Volume with 4 dimensions and the name is {}'.format(each_metric))

                # Accumulate all the scalar metric volumes one by one
                accum_all_output[each_method] = method_accum_vol

            # Extract Voxel Indices
            true_vox_inds = np.where(mask_vol == 1)
            true_vox_inds = np.asarray(true_vox_inds)
            true_vox_inds = np.transpose(true_vox_inds)
            len_true_vox = len(true_vox_inds)

            current_retrieval = 0
            while current_retrieval < n_retrievals:
                # initialize our batches of images and labels
                images = np.empty((bs, patch_size[0], patch_size[1], patch_size[2], n_classes))

                labels_bs = np.empty((bs, 6))
                labels_ivim = np.empty((bs, 3))
                labels_mcsmt = np.empty((bs, 2))
                labels_noddi = np.empty((bs, 6))
                labels_dti = np.empty((bs, 2))
                labels_mtcsd = np.empty((bs, 45))

                # X = np.empty((self.batch_size, self.n_classes))
                # y = np.empty((self.batch_size, self.n_classes))

                # Generate Random Inds for usage
                rand_inds = random.sample(range(len_true_vox - 1), bs)

                # Generate data
                for each_ind, each in enumerate(rand_inds):
                    # Retrieve indices
                    vox_inds = true_vox_inds[each, :]
                    images[each_ind, :] = np.squeeze(input_vol[vox_inds[0]-1:vox_inds[0]+2,
                                                               vox_inds[1]-1:vox_inds[1]+2,
                                                               vox_inds[2]-1:vox_inds[2]+2, :])

                    # Compartmental Models
                    labels_bs[each_ind, :] = np.squeeze(
                        accum_all_output['BS_2003'][vox_inds[0], vox_inds[1], vox_inds[2], :])

                    labels_ivim[each_ind, :] = np.squeeze(
                        accum_all_output['IVIM'][vox_inds[0], vox_inds[1], vox_inds[2], :])

                    labels_mcsmt[each_ind, :] = np.squeeze(
                        accum_all_output['MC_SMT'][vox_inds[0], vox_inds[1], vox_inds[2], :])

                    labels_noddi[each_ind, :] = np.squeeze(
                        accum_all_output['NODDI_WATSON'][vox_inds[0], vox_inds[1], vox_inds[2], :])

                    labels_dti[each_ind, :] = np.squeeze(
                        accum_all_output['DTI'][vox_inds[0], vox_inds[1], vox_inds[2], :])

                    # Orientation Models
                    labels_mtcsd[each_ind, :] = np.squeeze(
                        accum_all_output['MT_CSD'][vox_inds[0], vox_inds[1], vox_inds[2], :])

                current_retrieval = current_retrieval + bs
                yield (images, {'ball_stick': labels_bs,
                                'ivim': labels_ivim,
                                'mc_smt':labels_mcsmt,
                                'noddi':labels_noddi,
                                'dti': labels_dti,
                                'mt_csd': labels_mtcsd})

def nifti_compart_dti_generator(inputPath, bs, patch_size):
    # open the CSV file for reading
    #f = open(inputPath, "r")
    n_retrievals = 250000
    n_classes = 45
    # loop indefinitely
    while True:

        for vol_index, each_vol in enumerate(inputPath):

            # Load Nifti Volumes of Input, Output and Mask
            input_vol = load_nifty(each_vol['input_image'], data_type='float32')
            mask_vol = load_nifty(each_vol['mask'], data_type='float32')
            dims = mask_vol.shape
            # Convert mask_vol to int to save space
            mask_vol = np.int16(mask_vol)

            # TODO Output volumes since there are many, this needs to be handled a bit carefully
            # TODO The output object is a dictionary and contains 4 models by keys of
            # TODO 'BS_2003', 'IVIM', 'MC_SMT', 'NODDI_WATSON' (The Ordering is important)
            output_object = each_vol['output']
            ms_methods_list = ['BS_2003', 'IVIM', 'MC_SMT', 'NODDI_WATSON', 'DTI']

            # Accumulation of all metrics for all methods in a dictionary by name
            accum_all_output = {}

            for each_method in ms_methods_list:

                print('Reading Files of Method: {}'.format(each_method))

                # Grab list of ms metrics for the method
                metric_list = output_object[each_method]
                num_metrics = len(metric_list)

                # Create an accumulated empty volume for metrics
                method_accum_vol = np.zeros((dims[0], dims[1], dims[2], num_metrics))

                # Accumulate by looping over the metrics
                for metric_index, each_metric in enumerate(metric_list):
                    print('File name being loaded ... {}'.format(each_metric))
                    t_vol = load_nifty(each_metric, data_type='float32')

                    if len(t_vol.shape) == 3:
                        method_accum_vol[:, :, :, metric_index] = t_vol
                    else:
                        print('Found a Volume with 4 dimensions and the name is {}'.format(each_metric))

                # Accumulate all the scalar metric volumes one by one
                accum_all_output[each_method] = method_accum_vol

            # Extract Voxel Indices
            true_vox_inds = np.where(mask_vol == 1)
            true_vox_inds = np.asarray(true_vox_inds)
            true_vox_inds = np.transpose(true_vox_inds)
            len_true_vox = len(true_vox_inds)

            current_retrieval = 0
            while current_retrieval < n_retrievals:
                # initialize our batches of images and labels
                images = np.empty((bs, patch_size[0], patch_size[1], patch_size[2], n_classes))

                labels_bs = np.empty((bs, 6))
                labels_ivim = np.empty((bs, 3))
                labels_mcsmt = np.empty((bs, 2))
                labels_noddi = np.empty((bs, 6))
                labels_dti = np.empty((bs, 2))

                # X = np.empty((self.batch_size, self.n_classes))
                # y = np.empty((self.batch_size, self.n_classes))

                # Generate Random Inds for usage
                rand_inds = random.sample(range(len_true_vox - 1), bs)

                # Generate data
                for each_ind, each in enumerate(rand_inds):
                    # Retrieve indices
                    vox_inds = true_vox_inds[each, :]
                    images[each_ind, :] = np.squeeze(input_vol[vox_inds[0]-1:vox_inds[0]+2,
                                                               vox_inds[1]-1:vox_inds[1]+2,
                                                               vox_inds[2]-1:vox_inds[2]+2, :])

                    # Compartmental Models
                    labels_bs[each_ind, :] = np.squeeze(
                        accum_all_output['BS_2003'][vox_inds[0], vox_inds[1], vox_inds[2], :])

                    labels_ivim[each_ind, :] = np.squeeze(
                        accum_all_output['IVIM'][vox_inds[0], vox_inds[1], vox_inds[2], :])

                    labels_mcsmt[each_ind, :] = np.squeeze(
                        accum_all_output['MC_SMT'][vox_inds[0], vox_inds[1], vox_inds[2], :])

                    labels_noddi[each_ind, :] = np.squeeze(
                        accum_all_output['NODDI_WATSON'][vox_inds[0], vox_inds[1], vox_inds[2], :])

                    labels_dti[each_ind, :] = np.squeeze(
                        accum_all_output['DTI'][vox_inds[0], vox_inds[1], vox_inds[2], :])


                current_retrieval = current_retrieval + bs
                yield (images, {'ball_stick': labels_bs,
                                'ivim': labels_ivim,
                                'mc_smt':labels_mcsmt,
                                'noddi':labels_noddi,
                                'dti': labels_dti
                                }
                       )

def nifti_just_dti_generator(inputPath, bs, patch_size):
    # open the CSV file for reading
    #f = open(inputPath, "r")
    n_retrievals = 25000
    n_classes = 45
    # loop indefinitely
    while True:

        for vol_index, each_vol in enumerate(inputPath):

            # Load Nifti Volumes of Input, Output and Mask
            input_vol = load_nifty(each_vol['input_image'], data_type='float32')
            mask_vol = load_nifty(each_vol['mask'], data_type='float32')
            dims = mask_vol.shape
            # Convert mask_vol to int to save space
            mask_vol = np.int16(mask_vol)

            # TODO Output volumes since there are many, this needs to be handled a bit carefully
            # TODO The output object is a dictionary and contains 4 models by keys of
            # TODO 'BS_2003', 'IVIM', 'MC_SMT', 'NODDI_WATSON' (The Ordering is important)
            output_object = each_vol['output']
            ms_methods_list = ['DTI']

            # Accumulation of all metrics for all methods in a dictionary by name
            accum_all_output = {}

            for each_method in ms_methods_list:

                print('Reading Files of Method: {}'.format(each_method))

                # Grab list of ms metrics for the method
                metric_list = output_object[each_method]
                num_metrics = len(metric_list)

                # Create an accumulated empty volume for metrics
                method_accum_vol = np.zeros((dims[0], dims[1], dims[2], num_metrics))

                # Accumulate by looping over the metrics
                for metric_index, each_metric in enumerate(metric_list):
                    print('File name being loaded ... {}'.format(each_metric))
                    t_vol = load_nifty(each_metric, data_type='float32')

                    if len(t_vol.shape) == 3:
                        method_accum_vol[:, :, :, metric_index] = t_vol
                    else:
                        print('Found a Volume with 4 dimensions and the name is {}'.format(each_metric))

                # Accumulate all the scalar metric volumes one by one
                accum_all_output[each_method] = method_accum_vol

            # Extract Voxel Indices
            true_vox_inds = np.where(mask_vol == 1)
            true_vox_inds = np.asarray(true_vox_inds)
            true_vox_inds = np.transpose(true_vox_inds)
            len_true_vox = len(true_vox_inds)

            current_retrieval = 0
            while current_retrieval < n_retrievals:
                # initialize our batches of images and labels
                images = np.empty((bs, patch_size[0], patch_size[1], patch_size[2], n_classes))

                #labels_bs = np.empty((bs, 6))
                #labels_ivim = np.empty((bs, 3))
                #labels_mcsmt = np.empty((bs, 2))
                #labels_noddi = np.empty((bs, 6))
                labels_dti = np.empty((bs, 2))

                # X = np.empty((self.batch_size, self.n_classes))
                # y = np.empty((self.batch_size, self.n_classes))

                # Generate Random Inds for usage
                rand_inds = random.sample(range(len_true_vox - 1), bs)

                # Generate data
                for each_ind, each in enumerate(rand_inds):
                    # Retrieve indices
                    vox_inds = true_vox_inds[each, :]
                    images[each_ind, :] = np.squeeze(input_vol[vox_inds[0]-1:vox_inds[0]+2,
                                                               vox_inds[1]-1:vox_inds[1]+2,
                                                               vox_inds[2]-1:vox_inds[2]+2, :])

                    # Compartmental Models
                    #labels_bs[each_ind, :] = np.squeeze(
                    #    accum_all_output['BS_2003'][vox_inds[0], vox_inds[1], vox_inds[2], :])

                    #labels_ivim[each_ind, :] = np.squeeze(
                    #    accum_all_output['IVIM'][vox_inds[0], vox_inds[1], vox_inds[2], :])

                    #labels_mcsmt[each_ind, :] = np.squeeze(
                    #    accum_all_output['MC_SMT'][vox_inds[0], vox_inds[1], vox_inds[2], :])

                    #labels_noddi[each_ind, :] = np.squeeze(
                    #    accum_all_output['NODDI_WATSON'][vox_inds[0], vox_inds[1], vox_inds[2], :])

                    labels_dti[each_ind, :] = np.squeeze(
                        accum_all_output['DTI'][vox_inds[0], vox_inds[1], vox_inds[2], :])


                current_retrieval = current_retrieval + bs
                yield (images, {'dti': labels_dti})

def nifti_just_mcsmt_generator(inputPath, bs, patch_size):
    # open the CSV file for reading
    #f = open(inputPath, "r")
    n_retrievals = 50000
    n_classes = 45
    # loop indefinitely
    while True:

        for vol_index, each_vol in enumerate(inputPath):

            # Load Nifti Volumes of Input, Output and Mask
            input_vol = load_nifty(each_vol['input_image'], data_type='float32')
            mask_vol = load_nifty(each_vol['mask'], data_type='float32')
            dims = mask_vol.shape
            # Convert mask_vol to int to save space
            mask_vol = np.int16(mask_vol)

            # TODO Output volumes since there are many, this needs to be handled a bit carefully
            # TODO The output object is a dictionary and contains 4 models by keys of
            # TODO 'BS_2003', 'IVIM', 'MC_SMT', 'NODDI_WATSON' (The Ordering is important)
            output_object = each_vol['output']
            ms_methods_list = ['MC_SMT']

            # Accumulation of all metrics for all methods in a dictionary by name
            accum_all_output = {}

            for each_method in ms_methods_list:

                print('Reading Files of Method: {}'.format(each_method))

                # Grab list of ms metrics for the method
                metric_list = output_object[each_method]
                num_metrics = len(metric_list)

                # Create an accumulated empty volume for metrics
                method_accum_vol = np.zeros((dims[0], dims[1], dims[2], num_metrics))

                # Accumulate by looping over the metrics
                for metric_index, each_metric in enumerate(metric_list):
                    print('File name being loaded ... {}'.format(each_metric))
                    t_vol = load_nifty(each_metric, data_type='float32')

                    if len(t_vol.shape) == 3:
                        method_accum_vol[:, :, :, metric_index] = t_vol
                    else:
                        print('Found a Volume with 4 dimensions and the name is {}'.format(each_metric))

                # Accumulate all the scalar metric volumes one by one
                accum_all_output[each_method] = method_accum_vol

            # Extract Voxel Indices
            true_vox_inds = np.where(mask_vol == 1)
            true_vox_inds = np.asarray(true_vox_inds)
            true_vox_inds = np.transpose(true_vox_inds)
            len_true_vox = len(true_vox_inds)

            current_retrieval = 0
            while current_retrieval < n_retrievals:
                # initialize our batches of images and labels
                images = np.empty((bs, patch_size[0], patch_size[1], patch_size[2], n_classes))

                #labels_bs = np.empty((bs, 6))
                #labels_ivim = np.empty((bs, 3))
                labels_mcsmt_1 = np.empty((bs, 1))
                labels_mcsmt_2 = np.empty((bs, 1))
                #labels_noddi = np.empty((bs, 6))
                #labels_dti = np.empty((bs, 2))

                # X = np.empty((self.batch_size, self.n_classes))
                # y = np.empty((self.batch_size, self.n_classes))

                # Generate Random Inds for usage
                rand_inds = random.sample(range(len_true_vox - 1), bs)

                # Generate data
                for each_ind, each in enumerate(rand_inds):
                    # Retrieve indices
                    vox_inds = true_vox_inds[each, :]
                    images[each_ind, :] = np.squeeze(input_vol[vox_inds[0]-1:vox_inds[0]+2,
                                                               vox_inds[1]-1:vox_inds[1]+2,
                                                               vox_inds[2]-1:vox_inds[2]+2, :])

                    # Compartmental Models
                    #labels_bs[each_ind, :] = np.squeeze(
                    #    accum_all_output['BS_2003'][vox_inds[0], vox_inds[1], vox_inds[2], :])

                    #labels_ivim[each_ind, :] = np.squeeze(
                    #    accum_all_output['IVIM'][vox_inds[0], vox_inds[1], vox_inds[2], :])

                    labels_mcsmt_1[each_ind, :] = np.squeeze(
                        accum_all_output['MC_SMT'][vox_inds[0], vox_inds[1], vox_inds[2], 0])

                    labels_mcsmt_2[each_ind, :] = np.squeeze(
                        accum_all_output['MC_SMT'][vox_inds[0], vox_inds[1], vox_inds[2], 1])

                    #labels_noddi[each_ind, :] = np.squeeze(
                    #    accum_all_output['NODDI_WATSON'][vox_inds[0], vox_inds[1], vox_inds[2], :])

                    #labels_dti[each_ind, :] = np.squeeze(
                    #    accum_all_output['DTI'][vox_inds[0], vox_inds[1], vox_inds[2], :])


                current_retrieval = current_retrieval + bs
                yield (images, {'mc_smt_1': labels_mcsmt_1, 'mc_smt_2': labels_mcsmt_2})


def nifti_just_ivim_generator(inputPath, bs, patch_size):
    # open the CSV file for reading
    #f = open(inputPath, "r")
    n_retrievals = 25000
    n_classes = 45
    # loop indefinitely
    while True:

        for vol_index, each_vol in enumerate(inputPath):

            # Load Nifti Volumes of Input, Output and Mask
            input_vol = load_nifty(each_vol['input_image'], data_type='float32')
            mask_vol = load_nifty(each_vol['mask'], data_type='float32')
            dims = mask_vol.shape
            # Convert mask_vol to int to save space
            mask_vol = np.int16(mask_vol)

            # TODO Output volumes since there are many, this needs to be handled a bit carefully
            # TODO The output object is a dictionary and contains 4 models by keys of
            # TODO 'BS_2003', 'DTI', 'MC_SMT', 'NODDI_WATSON' (The Ordering is important)
            output_object = each_vol['output']
            ms_methods_list = ['IVIM']

            # Accumulation of all metrics for all methods in a dictionary by name
            accum_all_output = {}

            for each_method in ms_methods_list:

                print('Reading Files of Method: {}'.format(each_method))

                # Grab list of ms metrics for the method
                metric_list = output_object[each_method]
                num_metrics = len(metric_list)

                # Create an accumulated empty volume for metrics
                method_accum_vol = np.zeros((dims[0], dims[1], dims[2], num_metrics))

                # Accumulate by looping over the metrics
                for metric_index, each_metric in enumerate(metric_list):
                    print('File name being loaded ... {}'.format(each_metric))
                    t_vol = load_nifty(each_metric, data_type='float32')

                    if len(t_vol.shape) == 3:
                        method_accum_vol[:, :, :, metric_index] = t_vol
                    else:
                        print('Found a Volume with 4 dimensions and the name is {}'.format(each_metric))

                # Accumulate all the scalar metric volumes one by one
                accum_all_output[each_method] = method_accum_vol

            # Extract Voxel Indices
            true_vox_inds = np.where(mask_vol == 1)
            true_vox_inds = np.asarray(true_vox_inds)
            true_vox_inds = np.transpose(true_vox_inds)
            len_true_vox = len(true_vox_inds)

            current_retrieval = 0
            while current_retrieval < n_retrievals:
                # initialize our batches of images and labels
                images = np.empty((bs, patch_size[0], patch_size[1], patch_size[2], n_classes))

                #labels_bs = np.empty((bs, 6))
                labels_ivim = np.empty((bs, 3))
                #labels_mcsmt = np.empty((bs, 2))
                #labels_noddi = np.empty((bs, 6))
                #labels_dti = np.empty((bs, 2))

                # X = np.empty((self.batch_size, self.n_classes))
                # y = np.empty((self.batch_size, self.n_classes))

                # Generate Random Inds for usage
                rand_inds = random.sample(range(len_true_vox - 1), bs)

                # Generate data
                for each_ind, each in enumerate(rand_inds):
                    # Retrieve indices
                    vox_inds = true_vox_inds[each, :]
                    images[each_ind, :] = np.squeeze(input_vol[vox_inds[0]-1:vox_inds[0]+2,
                                                               vox_inds[1]-1:vox_inds[1]+2,
                                                               vox_inds[2]-1:vox_inds[2]+2, :])

                    # Compartmental Models
                    #labels_bs[each_ind, :] = np.squeeze(
                    #    accum_all_output['BS_2003'][vox_inds[0], vox_inds[1], vox_inds[2], :])

                    labels_ivim[each_ind, :] = np.squeeze(
                        accum_all_output['IVIM'][vox_inds[0], vox_inds[1], vox_inds[2], :])

                    #labels_mcsmt[each_ind, :] = np.squeeze(
                    #    accum_all_output['MC_SMT'][vox_inds[0], vox_inds[1], vox_inds[2], :])

                    #labels_noddi[each_ind, :] = np.squeeze(
                    #    accum_all_output['NODDI_WATSON'][vox_inds[0], vox_inds[1], vox_inds[2], :])

                    #labels_dti[each_ind, :] = np.squeeze(
                    #    accum_all_output['DTI'][vox_inds[0], vox_inds[1], vox_inds[2], :])


                current_retrieval = current_retrieval + bs
                yield (images, {'ivim': labels_ivim})

def nifti_dti_ivim_generator(inputPath, bs, patch_size):
    # open the CSV file for reading
    #f = open(inputPath, "r")
    n_retrievals = 25000
    n_classes = 45
    # loop indefinitely
    while True:

        for vol_index, each_vol in enumerate(inputPath):

            # Load Nifti Volumes of Input, Output and Mask
            input_vol = load_nifty(each_vol['input_image'], data_type='float32')
            mask_vol = load_nifty(each_vol['mask'], data_type='float32')
            dims = mask_vol.shape
            # Convert mask_vol to int to save space
            mask_vol = np.int16(mask_vol)

            # TODO Output volumes since there are many, this needs to be handled a bit carefully
            # TODO The output object is a dictionary and contains 4 models by keys of
            # TODO 'BS_2003', 'DTI', 'MC_SMT', 'NODDI_WATSON' (The Ordering is important)
            output_object = each_vol['output']
            ms_methods_list = ['IVIM', 'DTI']

            # Accumulation of all metrics for all methods in a dictionary by name
            accum_all_output = {}

            for each_method in ms_methods_list:

                print('Reading Files of Method: {}'.format(each_method))

                # Grab list of ms metrics for the method
                metric_list = output_object[each_method]
                num_metrics = len(metric_list)

                # Create an accumulated empty volume for metrics
                method_accum_vol = np.zeros((dims[0], dims[1], dims[2], num_metrics))

                # Accumulate by looping over the metrics
                for metric_index, each_metric in enumerate(metric_list):
                    print('File name being loaded ... {}'.format(each_metric))
                    t_vol = load_nifty(each_metric, data_type='float32')

                    if len(t_vol.shape) == 3:
                        method_accum_vol[:, :, :, metric_index] = t_vol
                    else:
                        print('Found a Volume with 4 dimensions and the name is {}'.format(each_metric))

                # Accumulate all the scalar metric volumes one by one
                accum_all_output[each_method] = method_accum_vol

            # Extract Voxel Indices
            true_vox_inds = np.where(mask_vol == 1)
            true_vox_inds = np.asarray(true_vox_inds)
            true_vox_inds = np.transpose(true_vox_inds)
            len_true_vox = len(true_vox_inds)

            current_retrieval = 0
            while current_retrieval < n_retrievals:
                # initialize our batches of images and labels
                images = np.empty((bs, patch_size[0], patch_size[1], patch_size[2], n_classes))

                #labels_bs = np.empty((bs, 6))
                labels_ivim = np.empty((bs, 3))
                #labels_mcsmt = np.empty((bs, 2))
                #labels_noddi = np.empty((bs, 6))
                labels_dti = np.empty((bs, 2))

                # X = np.empty((self.batch_size, self.n_classes))
                # y = np.empty((self.batch_size, self.n_classes))

                # Generate Random Inds for usage
                rand_inds = random.sample(range(len_true_vox - 1), bs)

                # Generate data
                for each_ind, each in enumerate(rand_inds):
                    # Retrieve indices
                    vox_inds = true_vox_inds[each, :]
                    images[each_ind, :] = np.squeeze(input_vol[vox_inds[0]-1:vox_inds[0]+2,
                                                               vox_inds[1]-1:vox_inds[1]+2,
                                                               vox_inds[2]-1:vox_inds[2]+2, :])

                    # Compartmental Models
                    #labels_bs[each_ind, :] = np.squeeze(
                    #    accum_all_output['BS_2003'][vox_inds[0], vox_inds[1], vox_inds[2], :])

                    labels_ivim[each_ind, :] = np.squeeze(
                        accum_all_output['IVIM'][vox_inds[0], vox_inds[1], vox_inds[2], :])

                    #labels_mcsmt[each_ind, :] = np.squeeze(
                    #    accum_all_output['MC_SMT'][vox_inds[0], vox_inds[1], vox_inds[2], :])

                    #labels_noddi[each_ind, :] = np.squeeze(
                    #    accum_all_output['NODDI_WATSON'][vox_inds[0], vox_inds[1], vox_inds[2], :])

                    labels_dti[each_ind, :] = np.squeeze(
                        accum_all_output['DTI'][vox_inds[0], vox_inds[1], vox_inds[2], :])


                current_retrieval = current_retrieval + bs
                yield (images, {'ivim': labels_ivim,
                                'dti': labels_dti}
                       )