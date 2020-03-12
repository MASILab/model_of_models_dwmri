import os
import nibabel as nib
import numpy as np

base_path = r'/nfs/masi/nathv/cortical_dysplasia'
base_path = os.path.normpath(base_path)

mask_name = 'mask.nii.gz'
mask_path = os.path.join(base_path, mask_name)

# Read the mask file
mask_obj = nib.load(mask_path)
mask_data = mask_obj.get_data()

# Get Dimensions and clear the mask on the edges a bit
dims = mask_data.shape

z_slices = dims[2]

# Clear mask on edges
mask_data[:,:,0:2] = 0
mask_data[:,:,z_slices-2:z_slices] = 0

# Save a new nifti file

# Retrieve affine
affine = mask_obj.affine
new_mask = nib.Nifti1Image(mask_data, affine)
new_mask_name = 'brain_mask_clean.nii.gz'
new_mask_path = os.path.join(base_path, new_mask_name)
nib.save(new_mask, new_mask_path)