from dmipy.signal_models import gaussian_models
from dmipy.core import modeling_framework
from dmipy.data import saved_data

zeppelin = gaussian_models.G2Zeppelin()
smt_mod = modeling_framework.MultiCompartmentSphericalMeanModel(models=[zeppelin])
scheme_hcp, data_hcp = saved_data.wu_minn_hcp_coronal_slice()

sub_image = data_hcp[70:90,: , 70:90]

import matplotlib.pyplot as plt
import matplotlib.patches as patches

fig, ax = plt.subplots(1)
ax.imshow(data_hcp[:, 0, :, 0].T, origin=True)
rect = patches.Rectangle((70,70),20,20,linewidth=1,edgecolor='r',facecolor='none')
ax.add_patch(rect)
ax.set_axis_off()
ax.set_title('HCP coronal slice B0 with ROI');
plt.clf()
#plt.show()

# Fitting SMT is very fast, half the time is actually spent estimating the spherical mean of the data.
smt_fit_hcp = smt_mod.fit(scheme_hcp, data_hcp, Ns=30, mask=data_hcp[..., 0]>0, use_parallel_processing=False)

fitted_parameters = smt_fit_hcp.fitted_parameters

fig, axs = plt.subplots(1, len(fitted_parameters), figsize=[15, 15])
axs = axs.ravel()

for i, (name, values) in enumerate(fitted_parameters.items()):
    cf = axs[i].imshow(values.squeeze().T, origin=True)
    axs[i].set_title(name)
    axs[i].set_axis_off()
    fig.colorbar(cf, ax=axs[i], shrink=0.2)

plt.show()
print('Debug here')
