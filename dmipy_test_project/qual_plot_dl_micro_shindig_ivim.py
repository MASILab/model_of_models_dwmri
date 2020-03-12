import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

## Paths of the data
## Ground Truth
bs_gt_path = r'/nfs/masi/nathv/miccai_2020/micro_methods_hcp_mini/125525/IVIM'
bs_gt_path = os.path.normpath(bs_gt_path)

## Predictions
base_preds_path = r'/nfs/masi/nathv/miccai_2020/first_shindig_predictions/125525/predicted_volumes'
base_preds_path = os.path.normpath(base_preds_path)

## Path Constructions
# Predictions
bs_preds_path = os.path.join(base_preds_path, 'ivim.nii.gz')

# Ground Truth
bs_liso_path = os.path.join(bs_gt_path, 'G1Ball_1_lambda_iso_norm.nii.gz')
bs_pv0_path = os.path.join(bs_gt_path, 'partial_volume_0.nii.gz')
bs_pv1_path = os.path.join(bs_gt_path, 'partial_volume_1_norm.nii.gz')

# Load Data
bs_pred_obj = nib.load(bs_preds_path)
bs_pred_data = bs_pred_obj.get_fdata()

bs_liso_obj = nib.load(bs_liso_path)
bs_liso_data = bs_liso_obj.get_fdata()

bs_pv0_obj = nib.load(bs_pv0_path)
bs_pv0_data = bs_pv0_obj.get_fdata()

bs_pv1_obj = nib.load(bs_pv1_path)
bs_pv1_data = bs_pv1_obj.get_fdata()

# Predictions
plt.figure(1, figsize=(12, 10))
plt.subplot(2,3,1)
plt.imshow(np.squeeze(bs_pred_data[:,:,70,0]))
plt.colorbar()
plt.clim(0,1)

plt.subplot(2,3,2)
plt.imshow(np.squeeze(bs_pred_data[:,:,70,1]))
plt.colorbar()
plt.clim(0,1)

plt.subplot(2,3,3)
plt.imshow(np.squeeze(bs_pred_data[:,:,70,2]))
plt.colorbar()
plt.clim(0,1)

# Ground Truths
plt.subplot(2,3,4)
plt.imshow(np.squeeze(bs_liso_data[:,:,70]))
plt.colorbar()
plt.clim(0,1)
plt.title('lambda iso')

plt.subplot(2,3,5)
plt.imshow(np.squeeze(bs_pv1_data[:,:,70]))
plt.colorbar()
plt.clim(0,1)
plt.title('partial vol 1')

plt.subplot(2,3,6)
plt.imshow(np.squeeze(bs_pv0_data[:,:,70]))
plt.colorbar()
plt.clim(0,1)
plt.title('partial vol 0')

plt.show()
print('Debug here')