"""
*Preliminary* pytorch implementation.
data generators for voxelmorph
"""

import numpy as np
import sys



def example_gen(ref_vol_names, mov_vol_names,batch_size=1):
    """
    generate examples
    Parameters:
        vol_names: a list or tuple of filenames
        batch_size: the size of the batch (default: 1)
        The following are fairly specific to our data structure, please change to your own
        return_segs: logical on whether to return segmentations
        seg_dir: the segmentations directory.
    """

    while True:
        idxes = np.random.randint(len(ref_vol_names), size=batch_size)

        X_data = []
        for idx in idxes:
            X = np.load(ref_vol_names[idx])
            X = X[np.newaxis, ..., np.newaxis]
            X_data.append(X)
        Y_data = []
        for idx in idxes:
            Y = np.load(mov_vol_names[idx])
            Y = Y[np.newaxis, ..., np.newaxis]
            Y_data.append(Y)
        print('learning:',ref_vol_names[idx],mov_vol_names[idx])

        if batch_size > 1:
            return_refs = np.concatenate(X_data, 0)
            return_movs = np.concatenate(Y_data, 0)
        else:
            return_refs = X_data[0]
            return_movs = np.concatenate(Y_data, 0)
        return return_refs,return_movs


if __name__ =='__main__':
    import glob
    import os
    import random
    batch_size = 5
    data_dir= 'D:\peizhunsd\data\\ref\\'
    data_dir1 = 'D:\peizhunsd\data\\mov\\'
    ref_vol_names = glob.glob(os.path.join(data_dir, '*.npy'))
    mov_vol_names = glob.glob(os.path.join(data_dir1,'*npy'))
    x,y= example_gen(ref_vol_names, mov_vol_names,batch_size)

