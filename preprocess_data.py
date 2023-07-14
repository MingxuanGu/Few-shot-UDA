# path operation
import os
from pathlib import Path
# image processing
import cv2
import SimpleITK as sitk
from scipy.ndimage import zoom
# numpy import
import numpy as np

# local import
from utils.utils import crop_volume


def preprocess_volume(img_volume):
    """
    :param img_volume: A patient volume
    :return: applying CLAHE and Bilateral filter for contrast enhacnmeent and denoising
    """
    prepross_imgs = []
    for i in range(len(img_volume)):
        img = img_volume[i]
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
        cl1 = clahe.apply(img)
        prepross_imgs.append(cl1)
    return np.array(prepross_imgs)


def nii_to_png_mscmrseg(crop_size=224, preprocess=True, parent_folder='preprocess2.0', target_res=1.):
    """
    preprocess the mscmrseg dataset. bssfp and t2 images are saved to '(train/test)A', the labels are saved to
    '(train/test)Amask'. The lge images and labels are saved to the corresponding folder replacing 'A' with 'B'.
    The images will be resized to have a pixel spacing of (1, 1) at the (x, y) dimension, while keep that at the z dim
    the same. The images will be histogram equalized with cv2.
    :param parent_folder:
    :param preprocess:
    :param crop_size: The size of the cropped image
    :param target_res:
    :return:
    """
    modalities = ['bSSFP', 't2', 'lge']
    datas = ['labels', 'dataset']  # ['dataset', 'labels']
    pat_id_start, pat_id_end = 1, 46
    modal_dict = {'bSSFP': 'C0', 't2': 'T2', 'lge': 'LGE'}
    for modality in modalities:
        st = 'A' if ((modality == 'bSSFP') or (modality == 't2')) else 'B'
        modal_fn = modal_dict[modality]
        for pat_id in range(pat_id_start, pat_id_end):
            for data in datas:
                if data == 'labels':
                    label = 'mask'
                    manual_fn = '_manual'
                    order = 0
                else:
                    label = ''
                    manual_fn = ''
                    order = 3
                train_test = 'test' if pat_id < 6 else 'train'
                print(f"saving the {pat_id}st {modality} {data}")
                path = f"F:/data/mscmrseg/raw_data/{data}/patient{pat_id}_{modal_fn}{manual_fn}.nii.gz"
                print(f'read from {path}')
                vol = sitk.ReadImage(path)
                vol = sitk.Cast(sitk.RescaleIntensity(vol), sitk.sitkUInt8)
                spacing = vol.GetSpacing()
                vol = sitk.GetArrayFromImage(vol)
                vol = zoom(vol, (1, spacing[0] / target_res, spacing[1] / target_res), order=order, mode='nearest')
                vol = crop_volume(vol, crop_size//2)
                if preprocess and data == 'dataset':
                    vol = preprocess_volume(vol)
                l = 0
                for m in vol:
                    save_path = f'F:/data/mscmrseg/{parent_folder}/{train_test}{st}{label}/pat_{pat_id}_{modality}_{l}.png'
                    if not Path(save_path).parent.exists():
                        Path(save_path).parent.mkdir(parents=True)
                        print(str(Path(save_path).parent) + ' created.')
                    cv2.imwrite(filename=save_path, img=m)
                    l += 1
    print("finish")


if __name__ == '__main__':
    nii_to_png_mscmrseg(crop_size=224, preprocess=True, parent_folder='preprocess2.0', target_res=1.)