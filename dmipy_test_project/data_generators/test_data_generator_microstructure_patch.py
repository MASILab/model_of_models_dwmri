import os
import nibabel as nib
import numpy as np
import time

def load_nifty(path_to_file, data_type):

    path_to_file = os.path.normpath(path_to_file)
    nifti_data = nib.load(path_to_file)
    nifti_img = nifti_data.get_fdata(dtype=data_type)
    nifti_data.uncache()
    return nifti_img

def save_nifti(predicted_vol, path_to_nifti_header, file_name, saver_path):

    nib_img = nib.Nifti1Image(predicted_vol, nib.load(path_to_nifti_header).affine, nib.load(path_to_nifti_header).header)
    # Grab ID from path to header
    #f_name = path_to_nifti_header.split('\\')
    f_name = file_name
    f_path = os.path.join(saver_path, f_name + '.nii.gz')
    nib.save(nib_img, f_path)

def test_patch_predictor(dl_model, test_data, save_path):

    # An attempt to reduce inference time as the v1 takes over 15 minutes per volume...
    vol_saver_path = os.path.join(save_path, 'predicted_volumes')
    if os.path.exists(vol_saver_path) is False:
        os.mkdir(vol_saver_path)

    batch_size = 1000
    for vol_index, each_vol in enumerate(test_data):

        start_time = time.time()

        # Load Nifti Volumes of Input, Output and Mask
        input_vol = load_nifty(each_vol['input_image'], data_type='float32')
        # TODO Commenting, output vol as the code is just for predicting new input data
        #output_vol = load_nifty(each_vol['output_image'], data_type='float32')
        mask_vol = load_nifty(each_vol['mask'], data_type='float32')

        # Convert mask_vol to int to save space
        mask_vol = np.int16(mask_vol)

        vol_dims = mask_vol.shape

        pred_vol_bs = np.zeros((vol_dims[0], vol_dims[1], vol_dims[2], 5))
        pred_vol_ivim = np.zeros((vol_dims[0], vol_dims[1], vol_dims[2], 3))
        pred_vol_mcsmt = np.zeros((vol_dims[0], vol_dims[1], vol_dims[2], 2))
        pred_vol_noddi = np.zeros((vol_dims[0], vol_dims[1], vol_dims[2], 5))

        batch_collector = np.zeros((batch_size, 3, 3, 3, 45))
        batch_inds = np.zeros((batch_size, 3), dtype='int16')
        batch_counter = 0

        for x in range(vol_dims[0]):
            print(x)
            for y in range(vol_dims[1]):
                for z in range(vol_dims[2]):
                    if mask_vol[x, y, z] == 1:
                        #ip_voxel = np.squeeze(input_vol[x, y, z, :])
                        #ip_voxel = np.reshape(ip_voxel, [1, 45])

                        ip_voxel = np.squeeze(input_vol[x - 1:x + 2,
                                                        y - 1:y + 2,
                                                        z - 1:z + 2, :])

                        batch_collector[batch_counter, :, :, :, :] = ip_voxel
                        batch_inds[batch_counter, :] = [x, y, z]
                        batch_counter = batch_counter + 1

                        #ip_voxel = np.reshape(ip_voxel, [1, 3, 3, 3, 45])
                        if batch_counter == batch_size:
                            # Make Preds
                            batch_preds = dl_model.predict(batch_collector)

                            # Assign Predictions to Prediction Volume
                            for per_pred in range(len(batch_collector)):
                                ret_x = batch_inds[per_pred, 0]
                                ret_y = batch_inds[per_pred, 1]
                                ret_z = batch_inds[per_pred, 2]

                                pred_vol_bs[ret_x, ret_y, ret_z, :] = batch_preds[0][per_pred]
                                pred_vol_ivim[ret_x, ret_y, ret_z, :] = batch_preds[1][per_pred]
                                pred_vol_mcsmt[ret_x, ret_y, ret_z, :] = batch_preds[2][per_pred]
                                pred_vol_noddi[ret_x, ret_y, ret_z, :] = batch_preds[3][per_pred]

                            # Set counter to zero and reinitiate batch collector and batch inds to zeros
                            batch_collector = np.zeros((batch_size, 3, 3, 3, 45))
                            batch_inds = np.zeros((batch_size, 3), dtype='int16')
                            batch_counter = 0

        end_time = time.time()
        time_taken = end_time - start_time
        print('Predictions Completed for Vol {} & Time Taken was {} \n'.format(vol_index, time_taken))
        print('Saving predicted volume')

        save_nifti(pred_vol_bs, each_vol['input_image'], 'ball_stick', vol_saver_path)
        save_nifti(pred_vol_ivim, each_vol['input_image'], 'ivim', vol_saver_path)
        save_nifti(pred_vol_mcsmt, each_vol['input_image'], 'mc_smt', vol_saver_path)
        save_nifti(pred_vol_noddi, each_vol['input_image'], 'noddi', vol_saver_path)

        print('Predicted Volume Saved ... \n ')

def test_compart_orient_predictor(dl_model, test_data, save_path):

    # An attempt to reduce inference time as the v1 takes over 15 minutes per volume...
    vol_saver_path = os.path.join(save_path, 'predicted_volumes')
    if os.path.exists(vol_saver_path) is False:
        os.mkdir(vol_saver_path)

    batch_size = 1000
    for vol_index, each_vol in enumerate(test_data):

        start_time = time.time()

        # Load Nifti Volumes of Input, Output and Mask
        input_vol = load_nifty(each_vol['input_image'], data_type='float32')
        # TODO Commenting, output vol as the code is just for predicting new input data
        #output_vol = load_nifty(each_vol['output_image'], data_type='float32')
        mask_vol = load_nifty(each_vol['mask'], data_type='float32')

        # Convert mask_vol to int to save space
        mask_vol = np.int16(mask_vol)

        vol_dims = mask_vol.shape

        pred_vol_bs = np.zeros((vol_dims[0], vol_dims[1], vol_dims[2], 6))
        pred_vol_ivim = np.zeros((vol_dims[0], vol_dims[1], vol_dims[2], 3))
        pred_vol_mcsmt = np.zeros((vol_dims[0], vol_dims[1], vol_dims[2], 2))
        pred_vol_noddi = np.zeros((vol_dims[0], vol_dims[1], vol_dims[2], 6))
        pred_vol_dti = np.zeros((vol_dims[0], vol_dims[1], vol_dims[2], 2))

        pred_vol_mtcsd = np.zeros((vol_dims[0], vol_dims[1], vol_dims[2], 45))

        batch_collector = np.zeros((batch_size, 3, 3, 3, 45))
        batch_inds = np.zeros((batch_size, 3), dtype='int16')
        batch_counter = 0

        for x in range(vol_dims[0]):
            print(x)
            for y in range(vol_dims[1]):
                for z in range(vol_dims[2]):
                    if mask_vol[x, y, z] == 1:
                        #ip_voxel = np.squeeze(input_vol[x, y, z, :])
                        #ip_voxel = np.reshape(ip_voxel, [1, 45])

                        ip_voxel = np.squeeze(input_vol[x - 1:x + 2,
                                                        y - 1:y + 2,
                                                        z - 1:z + 2, :])

                        batch_collector[batch_counter, :, :, :, :] = ip_voxel
                        batch_inds[batch_counter, :] = [x, y, z]
                        batch_counter = batch_counter + 1

                        #ip_voxel = np.reshape(ip_voxel, [1, 3, 3, 3, 45])
                        if batch_counter == batch_size:
                            # Make Preds
                            batch_preds = dl_model.predict(batch_collector)

                            # Assign Predictions to Prediction Volume
                            for per_pred in range(len(batch_collector)):
                                ret_x = batch_inds[per_pred, 0]
                                ret_y = batch_inds[per_pred, 1]
                                ret_z = batch_inds[per_pred, 2]

                                pred_vol_bs[ret_x, ret_y, ret_z, :] = batch_preds[0][per_pred]
                                pred_vol_ivim[ret_x, ret_y, ret_z, :] = batch_preds[1][per_pred]
                                pred_vol_mcsmt[ret_x, ret_y, ret_z, :] = batch_preds[2][per_pred]
                                pred_vol_noddi[ret_x, ret_y, ret_z, :] = batch_preds[3][per_pred]
                                pred_vol_dti[ret_x, ret_y, ret_z, :] = batch_preds[4][per_pred]
                                pred_vol_mtcsd[ret_x, ret_y, ret_z, :] = batch_preds[5][per_pred]

                            # Set counter to zero and reinitiate batch collector and batch inds to zeros
                            batch_collector = np.zeros((batch_size, 3, 3, 3, 45))
                            batch_inds = np.zeros((batch_size, 3), dtype='int16')
                            batch_counter = 0

        end_time = time.time()
        time_taken = end_time - start_time
        print('Predictions Completed for Vol {} & Time Taken was {} \n'.format(vol_index, time_taken))
        print('Saving predicted volume')

        save_nifti(pred_vol_bs, each_vol['input_image'], 'ball_stick', vol_saver_path)
        save_nifti(pred_vol_ivim, each_vol['input_image'], 'ivim', vol_saver_path)
        save_nifti(pred_vol_mcsmt, each_vol['input_image'], 'mc_smt', vol_saver_path)
        save_nifti(pred_vol_noddi, each_vol['input_image'], 'noddi', vol_saver_path)
        save_nifti(pred_vol_dti, each_vol['input_image'], 'dti', vol_saver_path)
        save_nifti(pred_vol_mtcsd, each_vol['input_image'], 'mtcsd', vol_saver_path)

        print('Predicted Volume Saved ... \n ')

def test_compart_dti_predictor(dl_model, test_data, save_path):

    # An attempt to reduce inference time as the v1 takes over 15 minutes per volume...
    vol_saver_path = os.path.join(save_path, 'predicted_volumes')
    if os.path.exists(vol_saver_path) is False:
        os.mkdir(vol_saver_path)

    batch_size = 1000
    for vol_index, each_vol in enumerate(test_data):

        start_time = time.time()

        # Load Nifti Volumes of Input, Output and Mask
        input_vol = load_nifty(each_vol['input_image'], data_type='float32')
        # TODO Commenting, output vol as the code is just for predicting new input data
        #output_vol = load_nifty(each_vol['output_image'], data_type='float32')
        mask_vol = load_nifty(each_vol['mask'], data_type='float32')

        # Convert mask_vol to int to save space
        mask_vol = np.int16(mask_vol)

        vol_dims = mask_vol.shape

        pred_vol_bs = np.zeros((vol_dims[0], vol_dims[1], vol_dims[2], 6))
        pred_vol_ivim = np.zeros((vol_dims[0], vol_dims[1], vol_dims[2], 3))
        pred_vol_mcsmt = np.zeros((vol_dims[0], vol_dims[1], vol_dims[2], 2))
        pred_vol_noddi = np.zeros((vol_dims[0], vol_dims[1], vol_dims[2], 6))
        pred_vol_dti = np.zeros((vol_dims[0], vol_dims[1], vol_dims[2], 2))

        batch_collector = np.zeros((batch_size, 3, 3, 3, 45))
        batch_inds = np.zeros((batch_size, 3), dtype='int16')
        batch_counter = 0

        for x in range(vol_dims[0]):
            print(x)
            for y in range(vol_dims[1]):
                for z in range(vol_dims[2]):
                    if mask_vol[x, y, z] == 1:
                        #ip_voxel = np.squeeze(input_vol[x, y, z, :])
                        #ip_voxel = np.reshape(ip_voxel, [1, 45])

                        ip_voxel = np.squeeze(input_vol[x - 1:x + 2,
                                                        y - 1:y + 2,
                                                        z - 1:z + 2, :])

                        batch_collector[batch_counter, :, :, :, :] = ip_voxel
                        batch_inds[batch_counter, :] = [x, y, z]
                        batch_counter = batch_counter + 1

                        #ip_voxel = np.reshape(ip_voxel, [1, 3, 3, 3, 45])
                        if batch_counter == batch_size:
                            # Make Preds
                            batch_preds = dl_model.predict(batch_collector)

                            # Assign Predictions to Prediction Volume
                            for per_pred in range(len(batch_collector)):
                                ret_x = batch_inds[per_pred, 0]
                                ret_y = batch_inds[per_pred, 1]
                                ret_z = batch_inds[per_pred, 2]

                                pred_vol_bs[ret_x, ret_y, ret_z, :] = batch_preds[0][per_pred]
                                pred_vol_ivim[ret_x, ret_y, ret_z, :] = batch_preds[1][per_pred]
                                pred_vol_mcsmt[ret_x, ret_y, ret_z, :] = batch_preds[2][per_pred]
                                pred_vol_noddi[ret_x, ret_y, ret_z, :] = batch_preds[3][per_pred]
                                pred_vol_dti[ret_x, ret_y, ret_z, :] = batch_preds[4][per_pred]

                            # Set counter to zero and reinitiate batch collector and batch inds to zeros
                            batch_collector = np.zeros((batch_size, 3, 3, 3, 45))
                            batch_inds = np.zeros((batch_size, 3), dtype='int16')
                            batch_counter = 0

        end_time = time.time()
        time_taken = end_time - start_time
        print('Predictions Completed for Vol {} & Time Taken was {} \n'.format(vol_index, time_taken))
        print('Saving predicted volume')

        save_nifti(pred_vol_bs, each_vol['input_image'], 'ball_stick', vol_saver_path)
        save_nifti(pred_vol_ivim, each_vol['input_image'], 'ivim', vol_saver_path)
        save_nifti(pred_vol_mcsmt, each_vol['input_image'], 'mc_smt', vol_saver_path)
        save_nifti(pred_vol_noddi, each_vol['input_image'], 'noddi', vol_saver_path)
        save_nifti(pred_vol_dti, each_vol['input_image'], 'dti', vol_saver_path)

        print('Predicted Volume Saved ... \n ')

def test_just_dti_predictor(dl_model, test_data, save_path):

    # An attempt to reduce inference time as the v1 takes over 15 minutes per volume...
    vol_saver_path = os.path.join(save_path, 'predicted_volumes')
    if os.path.exists(vol_saver_path) is False:
        os.mkdir(vol_saver_path)

    batch_size = 1000
    for vol_index, each_vol in enumerate(test_data):

        start_time = time.time()

        subj_id = each_vol['input_image'].split(sep='/')[-2]
        subj_save_path = os.path.join(vol_saver_path, subj_id)
        if os.path.exists(subj_save_path) == False:
            os.mkdir(subj_save_path)

        # Load Nifti Volumes of Input, Output and Mask
        input_vol = load_nifty(each_vol['input_image'], data_type='float32')
        # TODO Commenting, output vol as the code is just for predicting new input data
        #output_vol = load_nifty(each_vol['output_image'], data_type='float32')
        mask_vol = load_nifty(each_vol['mask'], data_type='float32')

        # Convert mask_vol to int to save space
        mask_vol = np.int16(mask_vol)

        vol_dims = mask_vol.shape

        #pred_vol_bs = np.zeros((vol_dims[0], vol_dims[1], vol_dims[2], 6))
        #pred_vol_ivim = np.zeros((vol_dims[0], vol_dims[1], vol_dims[2], 3))
        #pred_vol_mcsmt = np.zeros((vol_dims[0], vol_dims[1], vol_dims[2], 2))
        #pred_vol_noddi = np.zeros((vol_dims[0], vol_dims[1], vol_dims[2], 6))
        pred_vol_dti = np.zeros((vol_dims[0], vol_dims[1], vol_dims[2], 2))

        batch_collector = np.zeros((batch_size, 3, 3, 3, 45))
        batch_inds = np.zeros((batch_size, 3), dtype='int16')
        batch_counter = 0

        for x in range(vol_dims[0]):
            print(x)
            for y in range(vol_dims[1]):
                for z in range(vol_dims[2]):
                    if mask_vol[x, y, z] == 1:
                        #ip_voxel = np.squeeze(input_vol[x, y, z, :])
                        #ip_voxel = np.reshape(ip_voxel, [1, 45])

                        ip_voxel = np.squeeze(input_vol[x - 1:x + 2,
                                                        y - 1:y + 2,
                                                        z - 1:z + 2, :])

                        batch_collector[batch_counter, :, :, :, :] = ip_voxel
                        batch_inds[batch_counter, :] = [x, y, z]
                        batch_counter = batch_counter + 1

                        #ip_voxel = np.reshape(ip_voxel, [1, 3, 3, 3, 45])
                        if batch_counter == batch_size:
                            # Make Preds
                            batch_preds = dl_model.predict(batch_collector)

                            # Assign Predictions to Prediction Volume
                            for per_pred in range(len(batch_collector)):
                                ret_x = batch_inds[per_pred, 0]
                                ret_y = batch_inds[per_pred, 1]
                                ret_z = batch_inds[per_pred, 2]

                                #pred_vol_bs[ret_x, ret_y, ret_z, :] = batch_preds[0][per_pred]
                                #pred_vol_ivim[ret_x, ret_y, ret_z, :] = batch_preds[1][per_pred]
                                #pred_vol_mcsmt[ret_x, ret_y, ret_z, :] = batch_preds[2][per_pred]
                                #pred_vol_noddi[ret_x, ret_y, ret_z, :] = batch_preds[3][per_pred]
                                pred_vol_dti[ret_x, ret_y, ret_z, :] = batch_preds[per_pred]

                            # Set counter to zero and reinitiate batch collector and batch inds to zeros
                            batch_collector = np.zeros((batch_size, 3, 3, 3, 45))
                            batch_inds = np.zeros((batch_size, 3), dtype='int16')
                            batch_counter = 0

        end_time = time.time()
        time_taken = end_time - start_time
        print('Predictions Completed for Vol {} & Time Taken was {} \n'.format(vol_index, time_taken))
        print('Saving predicted volume')

        #save_nifti(pred_vol_bs, each_vol['input_image'], 'ball_stick', vol_saver_path)
        #save_nifti(pred_vol_ivim, each_vol['input_image'], 'ivim', vol_saver_path)
        #save_nifti(pred_vol_mcsmt, each_vol['input_image'], 'mc_smt', vol_saver_path)
        #save_nifti(pred_vol_noddi, each_vol['input_image'], 'noddi', vol_saver_path)
        save_nifti(pred_vol_dti, each_vol['input_image'], 'dti', subj_save_path)

        print('Predicted Volume Saved ... \n ')

def test_just_mcsmt_predictor(dl_model, test_data, save_path):

    # An attempt to reduce inference time as the v1 takes over 15 minutes per volume...
    vol_saver_path = os.path.join(save_path, 'predicted_volumes')
    if os.path.exists(vol_saver_path) is False:
        os.mkdir(vol_saver_path)

    batch_size = 1000
    for vol_index, each_vol in enumerate(test_data):

        start_time = time.time()

        subj_id = each_vol['input_image'].split(sep='/')[-2]
        subj_save_path = os.path.join(vol_saver_path, subj_id)
        if os.path.exists(subj_save_path) == False:
            os.mkdir(subj_save_path)

        # Load Nifti Volumes of Input, Output and Mask
        input_vol = load_nifty(each_vol['input_image'], data_type='float32')
        # TODO Commenting, output vol as the code is just for predicting new input data
        #output_vol = load_nifty(each_vol['output_image'], data_type='float32')
        mask_vol = load_nifty(each_vol['mask'], data_type='float32')

        # Convert mask_vol to int to save space
        mask_vol = np.int16(mask_vol)

        vol_dims = mask_vol.shape

        #pred_vol_bs = np.zeros((vol_dims[0], vol_dims[1], vol_dims[2], 6))
        #pred_vol_ivim = np.zeros((vol_dims[0], vol_dims[1], vol_dims[2], 3))
        pred_vol_mcsmt = np.zeros((vol_dims[0], vol_dims[1], vol_dims[2], 2))
        #pred_vol_noddi = np.zeros((vol_dims[0], vol_dims[1], vol_dims[2], 6))
        #pred_vol_dti = np.zeros((vol_dims[0], vol_dims[1], vol_dims[2], 2))

        batch_collector = np.zeros((batch_size, 3, 3, 3, 45))
        batch_inds = np.zeros((batch_size, 3), dtype='int16')
        batch_counter = 0

        for x in range(vol_dims[0]):
            print(x)
            for y in range(vol_dims[1]):
                for z in range(vol_dims[2]):
                    if mask_vol[x, y, z] == 1:
                        #ip_voxel = np.squeeze(input_vol[x, y, z, :])
                        #ip_voxel = np.reshape(ip_voxel, [1, 45])

                        ip_voxel = np.squeeze(input_vol[x - 1:x + 2,
                                                        y - 1:y + 2,
                                                        z - 1:z + 2, :])

                        batch_collector[batch_counter, :, :, :, :] = ip_voxel
                        batch_inds[batch_counter, :] = [x, y, z]
                        batch_counter = batch_counter + 1

                        #ip_voxel = np.reshape(ip_voxel, [1, 3, 3, 3, 45])
                        if batch_counter == batch_size:
                            # Make Preds
                            batch_preds = dl_model.predict(batch_collector)

                            # Assign Predictions to Prediction Volume
                            for per_pred in range(len(batch_collector)):
                                ret_x = batch_inds[per_pred, 0]
                                ret_y = batch_inds[per_pred, 1]
                                ret_z = batch_inds[per_pred, 2]

                                #pred_vol_bs[ret_x, ret_y, ret_z, :] = batch_preds[0][per_pred]
                                #pred_vol_ivim[ret_x, ret_y, ret_z, :] = batch_preds[1][per_pred]
                                pred_vol_mcsmt[ret_x, ret_y, ret_z, 0] = batch_preds[0][per_pred]
                                pred_vol_mcsmt[ret_x, ret_y, ret_z, 1] = batch_preds[1][per_pred]

                                #pred_vol_noddi[ret_x, ret_y, ret_z, :] = batch_preds[3][per_pred]
                                #pred_vol_dti[ret_x, ret_y, ret_z, :] = batch_preds[per_pred]

                            # Set counter to zero and reinitiate batch collector and batch inds to zeros
                            batch_collector = np.zeros((batch_size, 3, 3, 3, 45))
                            batch_inds = np.zeros((batch_size, 3), dtype='int16')
                            batch_counter = 0

        end_time = time.time()
        time_taken = end_time - start_time
        print('Predictions Completed for Vol {} & Time Taken was {} \n'.format(vol_index, time_taken))
        print('Saving predicted volume')

        #save_nifti(pred_vol_bs, each_vol['input_image'], 'ball_stick', vol_saver_path)
        #save_nifti(pred_vol_ivim, each_vol['input_image'], 'ivim', vol_saver_path)
        save_nifti(pred_vol_mcsmt, each_vol['input_image'], 'mc_smt', subj_save_path)
        #save_nifti(pred_vol_noddi, each_vol['input_image'], 'noddi', vol_saver_path)
        #save_nifti(pred_vol_dti, each_vol['input_image'], 'dti', subj_save_path)

        print('Predicted Volume Saved ... \n ')

def test_just_ivim_predictor(dl_model, test_data, save_path):

    # An attempt to reduce inference time as the v1 takes over 15 minutes per volume...
    vol_saver_path = os.path.join(save_path, 'predicted_volumes')
    if os.path.exists(vol_saver_path) is False:
        os.mkdir(vol_saver_path)

    batch_size = 1000
    for vol_index, each_vol in enumerate(test_data):

        start_time = time.time()

        subj_id = each_vol['input_image'].split(sep='/')[-2]
        subj_save_path = os.path.join(vol_saver_path, subj_id)
        if os.path.exists(subj_save_path) == False:
            os.mkdir(subj_save_path)

        # Load Nifti Volumes of Input, Output and Mask
        input_vol = load_nifty(each_vol['input_image'], data_type='float32')
        # TODO Commenting, output vol as the code is just for predicting new input data
        #output_vol = load_nifty(each_vol['output_image'], data_type='float32')
        mask_vol = load_nifty(each_vol['mask'], data_type='float32')

        # Convert mask_vol to int to save space
        mask_vol = np.int16(mask_vol)

        vol_dims = mask_vol.shape

        #pred_vol_bs = np.zeros((vol_dims[0], vol_dims[1], vol_dims[2], 6))
        pred_vol_ivim = np.zeros((vol_dims[0], vol_dims[1], vol_dims[2], 3))
        #pred_vol_mcsmt = np.zeros((vol_dims[0], vol_dims[1], vol_dims[2], 2))
        #pred_vol_noddi = np.zeros((vol_dims[0], vol_dims[1], vol_dims[2], 6))
        #pred_vol_dti = np.zeros((vol_dims[0], vol_dims[1], vol_dims[2], 2))

        batch_collector = np.zeros((batch_size, 3, 3, 3, 45))
        batch_inds = np.zeros((batch_size, 3), dtype='int16')
        batch_counter = 0

        for x in range(vol_dims[0]):
            print(x)
            for y in range(vol_dims[1]):
                for z in range(vol_dims[2]):
                    if mask_vol[x, y, z] == 1:
                        #ip_voxel = np.squeeze(input_vol[x, y, z, :])
                        #ip_voxel = np.reshape(ip_voxel, [1, 45])

                        ip_voxel = np.squeeze(input_vol[x - 1:x + 2,
                                                        y - 1:y + 2,
                                                        z - 1:z + 2, :])

                        batch_collector[batch_counter, :, :, :, :] = ip_voxel
                        batch_inds[batch_counter, :] = [x, y, z]
                        batch_counter = batch_counter + 1

                        #ip_voxel = np.reshape(ip_voxel, [1, 3, 3, 3, 45])
                        if batch_counter == batch_size:
                            # Make Preds
                            batch_preds = dl_model.predict(batch_collector)

                            # Assign Predictions to Prediction Volume
                            for per_pred in range(len(batch_collector)):
                                ret_x = batch_inds[per_pred, 0]
                                ret_y = batch_inds[per_pred, 1]
                                ret_z = batch_inds[per_pred, 2]

                                #pred_vol_bs[ret_x, ret_y, ret_z, :] = batch_preds[0][per_pred]
                                #pred_vol_ivim[ret_x, ret_y, ret_z, :] = batch_preds[1][per_pred]
                                #pred_vol_mcsmt[ret_x, ret_y, ret_z, :] = batch_preds[2][per_pred]
                                #pred_vol_noddi[ret_x, ret_y, ret_z, :] = batch_preds[3][per_pred]
                                pred_vol_ivim[ret_x, ret_y, ret_z, :] = batch_preds[per_pred]

                            # Set counter to zero and reinitiate batch collector and batch inds to zeros
                            batch_collector = np.zeros((batch_size, 3, 3, 3, 45))
                            batch_inds = np.zeros((batch_size, 3), dtype='int16')
                            batch_counter = 0

        end_time = time.time()
        time_taken = end_time - start_time
        print('Predictions Completed for Vol {} & Time Taken was {} \n'.format(vol_index, time_taken))
        print('Saving predicted volume')

        #save_nifti(pred_vol_bs, each_vol['input_image'], 'ball_stick', vol_saver_path)
        save_nifti(pred_vol_ivim, each_vol['input_image'], 'ivim', vol_saver_path)
        #save_nifti(pred_vol_mcsmt, each_vol['input_image'], 'mc_smt', vol_saver_path)
        #save_nifti(pred_vol_noddi, each_vol['input_image'], 'noddi', vol_saver_path)
        #save_nifti(pred_vol_dti, each_vol['input_image'], 'dti', subj_save_path)

        print('Predicted Volume Saved ... \n ')


def test_dti_ivim_predictor(dl_model, test_data, save_path):

    # An attempt to reduce inference time as the v1 takes over 15 minutes per volume...
    vol_saver_path = os.path.join(save_path, 'predicted_volumes')
    if os.path.exists(vol_saver_path) is False:
        os.mkdir(vol_saver_path)

    batch_size = 1000
    for vol_index, each_vol in enumerate(test_data):

        start_time = time.time()

        subj_id = each_vol['input_image'].split(sep='/')[-2]
        subj_save_path = os.path.join(vol_saver_path, subj_id)
        if os.path.exists(subj_save_path) == False:
            os.mkdir(subj_save_path)

        # Load Nifti Volumes of Input, Output and Mask
        input_vol = load_nifty(each_vol['input_image'], data_type='float32')
        # TODO Commenting, output vol as the code is just for predicting new input data
        #output_vol = load_nifty(each_vol['output_image'], data_type='float32')
        mask_vol = load_nifty(each_vol['mask'], data_type='float32')

        # Convert mask_vol to int to save space
        mask_vol = np.int16(mask_vol)

        vol_dims = mask_vol.shape

        #pred_vol_bs = np.zeros((vol_dims[0], vol_dims[1], vol_dims[2], 6))
        pred_vol_ivim = np.zeros((vol_dims[0], vol_dims[1], vol_dims[2], 3))
        #pred_vol_mcsmt = np.zeros((vol_dims[0], vol_dims[1], vol_dims[2], 2))
        #pred_vol_noddi = np.zeros((vol_dims[0], vol_dims[1], vol_dims[2], 6))
        pred_vol_dti = np.zeros((vol_dims[0], vol_dims[1], vol_dims[2], 2))

        batch_collector = np.zeros((batch_size, 3, 3, 3, 45))
        batch_inds = np.zeros((batch_size, 3), dtype='int16')
        batch_counter = 0

        for x in range(vol_dims[0]):
            print(x)
            for y in range(vol_dims[1]):
                for z in range(vol_dims[2]):
                    if mask_vol[x, y, z] == 1:
                        #ip_voxel = np.squeeze(input_vol[x, y, z, :])
                        #ip_voxel = np.reshape(ip_voxel, [1, 45])

                        ip_voxel = np.squeeze(input_vol[x - 1:x + 2,
                                                        y - 1:y + 2,
                                                        z - 1:z + 2, :])

                        batch_collector[batch_counter, :, :, :, :] = ip_voxel
                        batch_inds[batch_counter, :] = [x, y, z]
                        batch_counter = batch_counter + 1

                        #ip_voxel = np.reshape(ip_voxel, [1, 3, 3, 3, 45])
                        if batch_counter == batch_size:
                            # Make Preds
                            batch_preds = dl_model.predict(batch_collector)

                            # Assign Predictions to Prediction Volume
                            for per_pred in range(len(batch_collector)):
                                ret_x = batch_inds[per_pred, 0]
                                ret_y = batch_inds[per_pred, 1]
                                ret_z = batch_inds[per_pred, 2]

                                #pred_vol_bs[ret_x, ret_y, ret_z, :] = batch_preds[0][per_pred]
                                #pred_vol_ivim[ret_x, ret_y, ret_z, :] = batch_preds[1][per_pred]
                                #pred_vol_mcsmt[ret_x, ret_y, ret_z, :] = batch_preds[2][per_pred]
                                #pred_vol_noddi[ret_x, ret_y, ret_z, :] = batch_preds[3][per_pred]
                                pred_vol_ivim[ret_x, ret_y, ret_z, :] = batch_preds[0][per_pred]
                                pred_vol_dti[ret_x, ret_y, ret_z, :] = batch_preds[1][per_pred]

                            # Set counter to zero and reinitiate batch collector and batch inds to zeros
                            batch_collector = np.zeros((batch_size, 3, 3, 3, 45))
                            batch_inds = np.zeros((batch_size, 3), dtype='int16')
                            batch_counter = 0

        end_time = time.time()
        time_taken = end_time - start_time
        print('Predictions Completed for Vol {} & Time Taken was {} \n'.format(vol_index, time_taken))
        print('Saving predicted volume')

        #save_nifti(pred_vol_bs, each_vol['input_image'], 'ball_stick', vol_saver_path)
        save_nifti(pred_vol_ivim, each_vol['input_image'], 'ivim', vol_saver_path)
        #save_nifti(pred_vol_mcsmt, each_vol['input_image'], 'mc_smt', vol_saver_path)
        #save_nifti(pred_vol_noddi, each_vol['input_image'], 'noddi', vol_saver_path)
        save_nifti(pred_vol_dti, each_vol['input_image'], 'dti', subj_save_path)

        print('Predicted Volume Saved ... \n ')