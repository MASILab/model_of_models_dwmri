from dmipy.signal_models import cylinder_models, gaussian_models
from dmipy.distributions.distribute_models import BundleModel
from dmipy.core import modeling_framework
from dmipy.data import saved_data
import matplotlib.pyplot as plt
import matplotlib.patches as patches

stick = cylinder_models.C1Stick()
zeppelin = gaussian_models.G2Zeppelin()
ball = gaussian_models.G1Ball()

bundle = BundleModel([stick, zeppelin])
bundle.set_tortuous_parameter('G2Zeppelin_1_lambda_perp',
    'C1Stick_1_lambda_par','partial_volume_0')
bundle.set_equal_parameter('G2Zeppelin_1_lambda_par', 'C1Stick_1_lambda_par')
bundle.set_fixed_parameter('G2Zeppelin_1_lambda_par', 1.7e-9)

smt_noddi_mod = modeling_framework.MultiCompartmentSphericalMeanModel(models=[bundle, ball])
smt_noddi_mod.parameter_names

# then we fix the isotropic diffusivity
smt_noddi_mod.set_fixed_parameter('G1Ball_1_lambda_iso', 3e-9)

scheme_hcp, data_hcp = saved_data.wu_minn_hcp_coronal_slice()
sub_image = data_hcp[70:90,: , 70:90]

fig, ax = plt.subplots(1)
ax.imshow(data_hcp[:, 0, :, 0].T, origin=True)
rect = patches.Rectangle((70,70),20,20,linewidth=1,edgecolor='r',facecolor='none')
ax.add_patch(rect)
ax.set_axis_off()
ax.set_title('HCP coronal slice B0 with ROI');
plt.show()
plt.clf()

# Fitting a spherical mean model is again very fast.
smt_noddi_fit = smt_noddi_mod.fit(scheme_hcp, data_hcp, mask=data_hcp[..., 0]>0, Ns=10)

fitted_parameters = smt_noddi_fit.fitted_parameters

fig, axs = plt.subplots(1, len(fitted_parameters), figsize=[15, 5])
axs = axs.ravel()

for i, (name, values) in enumerate(fitted_parameters.items()):
    cf = axs[i].imshow(values.squeeze().T, origin=True)
    axs[i].set_title(name)
    rect = patches.Rectangle((70,70),20,20,linewidth=5,edgecolor='r',facecolor='none')
    axs[i].add_patch(rect)
    axs[i].set_axis_off()
    fig.colorbar(cf, ax=axs[i], shrink=.95)



