import numpy as np
import nibabel as nib
import os
import matplotlib.pyplot as plt

# Dipy Imports
import dipy.reconst.shm as shm
from dipy.core.gradients import gradient_table
from dipy.io.gradients import read_bvals_bvecs

def main():
    # Base HCP Data Path
    base_data_path = r'/nfs/HCP/data'
    base_data_path = os.path.normpath(base_data_path)

    # Subject ID's list
    subj_ID_List = ['125525', '118225', '116726']
    # TODO When needed loop here over the ID list
    # Subject Save Path
    # subj_save_path = os.path.join(base_save_path, subj_ID)
    #if os.path.exists(subj_save_path) == False:
    #    os.mkdir(subj_save_path)

    # TODO For later the subject data, bval and bvec reading part can be put inside a function
    subj_data_path = os.path.join(base_data_path, subj_ID_List[0], 'T1w', 'Diffusion')

    # Read the Nifti file, bvals and bvecs
    subj_bvals_fpath = os.path.join(subj_data_path, 'bvals')
    subj_bvecs_fpath = os.path.join(subj_data_path, 'bvecs')

    bvals, bvecs = read_bvals_bvecs(subj_bvals_fpath, subj_bvecs_fpath)
    gtab = gradient_table(bvals, bvecs)


    print('Debug here')

    return None

if __name__=="__main__":
    main()