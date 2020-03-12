import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

## Paths of the data
## Ground Truth
bs_gt_path = r'/nfs/masi/nathv/miccai_2020/micro_methods_hcp_mini/125525/BS_2003'
bs_gt_path = os.path.normpath(bs_gt_path)

## Predictions
base_preds_path = r'/nfs/masi/nathv/miccai_2020/first_shindig_predictions/125525/predicted_volumes'
base_preds_path = os.path.normpath(base_preds_path)

## Path Constructions
# Predictions
bs_preds_path = os.path.join(base_preds_path, 'ball_stick.nii.gz')

# Ground Truth
bs_lpar_path = os.path.join(bs_gt_path, 'C1Stick_1_lambda_par_norm.nii.gz')
bs_mu_path = os.path.join(bs_gt_path, 'C1Stick_1_mu_norm.nii.gz')
bs_liso_path = os.path.join(bs_gt_path, 'G1Ball_1_lambda_iso_norm.nii.gz')
bs_pv0_path = os.path.join(bs_gt_path, 'partial_volume_0.nii.gz')
bs_pv1_path = os.path.join(bs_gt_path, 'partial_volume_1.nii.gz')

# Load Data
bs_pred_obj = nib.load(bs_preds_path)
bs_pred_data = bs_pred_obj.get_fdata()

bs_lpar_obj = nib.load(bs_lpar_path)
bs_lpar_data = bs_lpar_obj.get_fdata()

bs_mu_obj = nib.load(bs_mu_path)
bs_mu_data = bs_mu_obj.get_fdata()

bs_liso_obj = nib.load(bs_liso_path)
bs_liso_data = bs_liso_obj.get_fdata()

bs_pv0_obj = nib.load(bs_pv0_path)
bs_pv0_data = bs_pv0_obj.get_fdata()

bs_pv1_obj = nib.load(bs_pv1_path)
bs_pv1_data = bs_pv1_obj.get_fdata()

# Predictions
plt.figure(1, figsize=(16, 10))
plt.subplot(2,5,1)
plt.imshow(np.squeeze(bs_pred_data[:,:,70,0]))
plt.colorbar()
plt.clim(0,1)

plt.subplot(2,5,2)
plt.imshow(np.squeeze(bs_pred_data[:,:,70,1]))
plt.colorbar()
plt.clim(0,1)

plt.subplot(2,5,3)
plt.imshow(np.squeeze(bs_pred_data[:,:,70,2]))
plt.colorbar()
plt.clim(0,1)

plt.subplot(2,5,4)
plt.imshow(np.squeeze(bs_pred_data[:,:,70,3]))
plt.colorbar()
plt.clim(0,1)

plt.subplot(2,5,5)
plt.imshow(np.squeeze(bs_pred_data[:,:,70,4]))
plt.colorbar()
plt.clim(0,1)

# Ground Truths
plt.subplot(2,5,6)
plt.imshow(np.squeeze(bs_pv0_data[:,:,70]))
plt.colorbar()
plt.clim(0,1)
plt.title('Partial Vol 0')

plt.subplot(2,5,7)
plt.imshow(np.squeeze(bs_liso_data[:,:,70]))
plt.colorbar()
plt.clim(0,1)
plt.title('lambda iso')

plt.subplot(2,5,8)
plt.imshow(np.squeeze(bs_mu_data[:,:,70,0]))
plt.colorbar()
plt.clim(0,1)
plt.title('Mu')

plt.subplot(2,5,9)
plt.imshow(np.squeeze(bs_pv1_data[:,:,70]))
plt.colorbar()
plt.clim(0,1)
plt.title('Partial Vol 1')

plt.subplot(2,5,10)
plt.imshow(np.squeeze(bs_lpar_data[:,:,70]))
plt.colorbar()
plt.clim(0,1)
plt.title('lambda par')

plt.show()
print('Debug here')