import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import time
from scipy.ndimage import rotate

from dipy.core.sphere import Sphere, HemiSphere
from dipy.viz import window, actor
from dipy.core.gradients import gradient_table

def main():
    base_path = r'/nfs/HCP/data/664757/T1w/Diffusion/'
    base_path = os.path.normpath(base_path)

    start_time = time.time()
    nifti_babel_obj = nib.load(os.path.join(base_path, 'data.nii.gz'))
    nifti_data = nifti_babel_obj.get_fdata()
    end_time = time.time()
    print('Time taken to Load Data: {}'.format(end_time - start_time))

    bvals = np.loadtxt(os.path.join(base_path, 'bvals'))
    bvecs = np.loadtxt(os.path.join(base_path, 'bvecs'))
    subj_mask_babel = nib.load(os.path.join(base_path, 'nodif_brain_mask.nii.gz'))
    mask_data = subj_mask_babel.get_fdata()
    mask_data = np.array(mask_data, dtype='bool')
    inv_mask_data = np.invert(mask_data)

    # Extract indices from bvalues where 1000 and 0 are present
    idxs_1k = [i for i in range(len(bvals)) if (bvals[i] > 900 and bvals[i] < 1100)]
    idxs_2k = [i for i in range(len(bvals)) if (bvals[i] > 1900 and bvals[i] < 2100)]
    idxs_3k = [i for i in range(len(bvals)) if (bvals[i] > 2900 and bvals[i] < 3100)]
    idxs_b0 = [i for i in range(len(bvals)) if (bvals[i] < 50)]

    mean_b0 = nifti_data[:, :, :, idxs_b0]
    mean_b0 = np.nanmean(mean_b0, axis=3)

    # Remove all b0's from bvecs and bvals for clean formation of sphere's
    grad_idxs = np.concatenate([idxs_1k, idxs_2k, idxs_3k])
    grad_bvals = bvals[grad_idxs]
    grad_bvecs = bvecs[:, grad_idxs]

    hemi_sphere = HemiSphere(xyz=bvecs[:, idxs_1k].transpose())
    hemi_sphere_2 = HemiSphere(xyz=bvecs[:, idxs_2k].transpose())
    hemi_sphere_3 = HemiSphere(xyz=bvecs[:, idxs_3k].transpose())

    # Plot Different Gradient Direction Slices
    slice_num = 72
    norm_b1k = nifti_data[:, :, :, idxs_1k]
    norm_b2k = nifti_data[:, :, :, idxs_2k]
    norm_b3k = nifti_data[:, :, :, idxs_3k]

    for idx, each_vol in enumerate(idxs_1k):

        temp_1 = np.divide(norm_b1k[:, :, :, idx], mean_b0)
        temp_1[inv_mask_data] = 0
        norm_b1k[:, :, :, idx] = temp_1

        temp_1 = np.divide(norm_b2k[:, :, :, idx], mean_b0)
        temp_1[inv_mask_data] = 0
        norm_b2k[:, :, :, idx] = temp_1

        temp_1 = np.divide(norm_b3k[:, :, :, idx], mean_b0)
        temp_1[inv_mask_data] = 0
        norm_b3k[:, :, :, idx] = temp_1

    nan_map = np.isnan(norm_b1k)
    norm_b1k[nan_map] = 0

    nan_map = np.isnan(norm_b2k)
    norm_b2k[nan_map] = 0

    nan_map = np.isnan(norm_b3k)
    norm_b3k[nan_map] = 0

    plt.figure(1, figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(rotate(np.squeeze(norm_b1k[:, :, slice_num, 0]), 90), cmap='gray')
    plt.colorbar()
    plt.clim(0, 1)

    plt.subplot(1, 3, 2)
    plt.imshow(rotate(np.squeeze(norm_b2k[:, :, slice_num, 0]), 90), cmap='gray')
    plt.colorbar()
    plt.clim(0, 1)

    plt.subplot(1, 3, 3)
    plt.imshow(rotate(np.squeeze(norm_b3k[:, :, slice_num, 0]), 90), cmap='gray')
    plt.colorbar()
    plt.clim(0, 1)

    plt.show()
    print('Debug here')

    #ren = window.Renderer()
    #ren.SetBackground(1, 1, 1)

    #ren.add(actor.point(hemi_sphere.vertices, window.colors.green, point_radius=0.04))
    #ren.add(actor.point(hemi_sphere_2.vertices * 2, window.colors.red, point_radius=0.06, _opacity=0.8))
    #ren.add(actor.point(hemi_sphere_3.vertices * 4, window.colors.blue, point_radius=0.06, _opacity=0.6))
    #ren.add(actor.point(hemi_sphere.vertices * 5, window.colors.banana, point_radius=0.06, _opacity=0.4))
    #ren.add(actor.point(hemi_sphere_2.vertices * 6, window.colors.blue_violet, point_radius=0.2))
    #ren.add(actor.point(hemi_sphere_3.vertices * 7, window.colors.black, point_radius=0.2))

    #interactive = True
    #if interactive:
    #    window.show(ren)
    return None

if __name__=="__main__":
    main()
