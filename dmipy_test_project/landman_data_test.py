import numpy as np
import os
import matplotlib.pyplot as plt

from dipy.io.image import load_nifti, save_nifti
from dipy.io.gradients import read_bvals_bvecs
from dipy.core.gradients import gradient_table
from dipy.segment.mask import median_otsu
from dipy.reconst.dti import fractional_anisotropy, color_fa
from dipy.viz import window, actor
import dipy.reconst.dti as dti

from dipy.data import get_fnames

nifti_path = r'/nfs/masi/nathv/ses-c07r1/dwi/sub-3_ses-c07r1_dwi.nii.gz'
nifti_path = os.path.normpath(nifti_path)

bval_path = r'/nfs/masi/nathv/ses-c07r1/dwi/sub-3_ses-c07r1_dwi.bval'
bval_path = os.path.normpath(bval_path)

bvec_path = r'/nfs/masi/nathv/ses-c07r1/dwi/sub-3_ses-c07r1_dwi.bvec'
bvec_path = os.path.normpath(bvec_path)

print('Loading Nifti Data ...')
data, affine = load_nifti(nifti_path)
print('Data Loaded ...')
print('Data Shape is {}'.format(data.shape))

bvals, bvecs = read_bvals_bvecs(bval_path, bvec_path)
gtab = gradient_table(bvals, bvecs)

maskdata, mask = median_otsu(data, vol_idx=range(10, 50), median_radius=3,
                             numpass=1, autocrop=True, dilate=2)
print('maskdata.shape (%d, %d, %d, %d)' % maskdata.shape)

tenmodel = dti.TensorModel(gtab)
tenfit = tenmodel.fit(maskdata)

FA = fractional_anisotropy(tenfit.evals)
FA[np.isnan(FA)] = 0

save_nifti('/nfs/masi/nathv/tensor_fa.nii.gz', FA.astype(np.float32), affine)
save_nifti('/nfs/masi/nathv/tensor_evecs.nii.gz', tenfit.evecs.astype(np.float32), affine)
print('Debug here')

plt.figure(1, figsize=(15,8))
plt.subplot(1,3,1)
plt.imshow(np.squeeze(FA[:,:,40]))
plt.colorbar()

plt.subplot(1,3,2)
plt.imshow(np.squeeze(FA[:,:,45]))
plt.colorbar()

plt.subplot(1,3,3)
plt.imshow(np.squeeze(FA[:,:,50]))
plt.colorbar()
plt.show()
plt.savefig('/nfs/masi/nathv/fa_plots.png')
plt.clf()
plt.close(1)

RGB = color_fa(FA, tenfit.evecs)
print('Computing tensor ellipsoids in a part of the splenium of the CC')

from dipy.data import get_sphere
sphere = get_sphere('repulsion724')

# Enables/disables interactive visualization
interactive = False

ren = window.Renderer()

evals = tenfit.evals[:, :, :]
evecs = tenfit.evecs[:, :, :]

cfa = RGB[:, :, :]
cfa /= cfa.max()

ren.add(actor.tensor_slicer(evals, evecs, scalar_colors=cfa, sphere=sphere,
                            scale=0.3))

print('Saving illustration as tensor_ellipsoids.png')
window.record(ren, n_frames=1, out_path='tensor_ellipsoids.png',
              size=(600, 600))
if interactive:
    window.show(ren)
