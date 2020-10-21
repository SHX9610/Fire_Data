import os
import glob
import ants
import numpy as np


##将 3min向pre配准。1.获取文件路径；2.读取数据，数据格式为 ants.core.ants_image.ANTsImage；3.进行配准，方法为Affine；4.保存配准结果。
fix_path = 'E:\peizhunsd\LPBA40\\fixed.nii.gz'
move_path = 'E:\peizhunsd\LPBA40\\train\\S11.delineation.skullstripped.nii.gz'
save_path = 'reg_3min.nii.gz'
fix_img = ants.image_read(fix_path)
move_img = ants.image_read(move_path)
outs = ants.registration(fix_img,move_img,type_of_transforme = 'Affine')
reg_img = outs['warpedmovout']
ants.image_write(reg_img,save_path)