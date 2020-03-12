import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from dmipy.signal_models import cylinder_models, gaussian_models
from dmipy.distributions.distribute_models import BundleModel
from dmipy.core import modeling_framework
from dmipy.core.acquisition_scheme import acquisition_scheme_from_bvalues

## Data Paths Nifti, Bval, Bvec
b1k_nifti_path = r'/nfs/masi/schilkg1/TestData/Donahue_139198.05.01.15-07-31.WIP_GLYMPH_DTI_opt_32_b1000.01.nii'
b1k_bval_path = r'/nfs/masi/schilkg1/TestData/Donahue_139198.05.01.15-07-31.WIP_GLYMPH_DTI_opt_32_b1000.01.bval'
b1k_bvec_path = r'/nfs/masi/schilkg1/TestData/Donahue_139198.05.01.15-07-31.WIP_GLYMPH_DTI_opt_32_b1000.01.bvec'

b2k_nifti_path = r'/nfs/masi/schilkg1/TestData/Donahue_139198.04.01.15-04-51.WIP_GLYMPH_DTI_opt_56_b2000.01.nii'
b2k_bval_path = r'/nfs/masi/schilkg1/TestData/Donahue_139198.04.01.15-04-51.WIP_GLYMPH_DTI_opt_56_b2000.01.bval'
b2k_bvec_path = r'/nfs/masi/schilkg1/TestData/Donahue_139198.04.01.15-04-51.WIP_GLYMPH_DTI_opt_56_b2000.01.bvec'

## Reading Nifti DW-MRI Data. Bvals and Bvecs
# Large data, USE Uncache
b1k_data = nib.load(b1k_nifti_path)
b1k_fdata = b1k_data.get_fdata()

b2k_data = nib.load(b2k_nifti_path)
b2k_fdata = b2k_data.get_fdata()

b1k_bvals = np.loadtxt(b1k_bval_path)
b1k_bvecs = np.loadtxt(b1k_bvec_path)

b2k_bvals = np.loadtxt(b2k_bval_path)
b2k_bvecs = np.loadtxt(b2k_bvec_path)

# Concatenante Bvals, Bvecs and Data
all_bvecs = np.hstack((b1k_bvecs, b2k_bvecs))
all_bvals = np.hstack((b1k_bvals, b2k_bvals))
all_data = np.concatenate((b1k_fdata, b2k_fdata), axis=3)

# Prepare Dmipy Acquisition Scheme
# This is Dmipy nonsense, that if B-values are in s/mm^2 they need
# to be multiplied with 1e6 to be of the scale s/m^2

all_bvals = all_bvals * 1e6
all_bvecs = np.transpose(all_bvecs)

# The below line also takes in small delta and big delta.
# TODO Big Delta and small delta are not available quite often for the data.
acq_scheme = acquisition_scheme_from_bvalues(all_bvals, all_bvecs)

# We are ready to fit models
# Prepare SMT Model
zeppelin = gaussian_models.G2Zeppelin()
smt_mod = modeling_framework.MultiCompartmentSphericalMeanModel(models=[zeppelin])
#smt_mod.set_fractional_parameter()

# Fit SMT
smt_fit_hcp = smt_mod.fit(acq_scheme, all_data, Ns=30, mask=all_data[..., 0]>0, use_parallel_processing=False)

# TODO Use a model name with the dictionary for saving the file name for a specific subject per model.
print('Debug here')