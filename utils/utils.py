import numpy as np
import cv2
import os
from skimage import measure
from torch.nn import init
import torch
from utils.callbacks import ModelCheckPointCallback


def load_nii(img_path):
    """
    Function to load a 'nii' or 'nii.gz' file.

    Parameters
    ----------

    img_path: string
    String with the path of the 'nii' or 'nii.gz' image file name.

    Returns
    -------
    a numpy array of the image values, the affine transformation of the image, the header of the image.
    """
    import nibabel as nib
    nimg = nib.load(img_path)
    return nimg.get_data(), nimg.affine, nimg.header


def read_img(pat_id, img_len, file_path='../processed/', modality='lge'):
    images = []
    if modality == 'bssfp':
        folder = 'testA' if pat_id < 6 else 'trainA'
    else:
        folder = 'testB' if pat_id < 6 else 'trainB'
    modality = 'bSSFP' if modality == 'bssfp' else 'lge'
    for im in range(img_len):
        img = cv2.imread(os.path.join(file_path, "{}/pat_{}_{}_{}.png".format(folder, pat_id, modality, im)))
        images.append(img)
    return np.array(images)


def keep_largest_connected_components(mask):
    """
    Keeps only the largest connected components of each label for a segmentation mask.
    Args:
        mask: the image to be processed [B, C, ...]

    Returns:

    """
    num_channel = mask.shape[1]
    out_img = np.zeros(mask.shape, dtype=np.uint8)
    for struc_id in range(1, num_channel + 1):
        binary_img = mask == struc_id
        blobs = measure.label(binary_img, connectivity=1)
        props = measure.regionprops(blobs)
        if not props:
            continue
        area = [ele.area for ele in props]
        largest_blob_ind = np.argmax(area)
        largest_blob_label = props[largest_blob_ind].label

        out_img[blobs == largest_blob_label] = struc_id

    return out_img


def resize_volume(img_volume, w=288, h=288):
    """
    :param img_volume:
    :return:
    """
    img_res = []
    for im in img_volume:
        img_res.append(cv2.resize(im, dsize=(w, h), interpolation=cv2.INTER_AREA))

    return np.array(img_res)


def crop_volume(vol, crop_size=112):
    """
    :param vol:
    :return:
    """

    return np.array(vol[:,
                    int(vol.shape[1] / 2) - crop_size: int(vol.shape[1] / 2) + crop_size,
                    int(vol.shape[2] / 2) - crop_size: int(vol.shape[2] / 2) + crop_size])


def reconstruct_volume(vol, crop_size=112, origin_size=256):
    """
    :param vol:
    :return:
    """
    recon_vol = np.zeros((len(vol), origin_size, origin_size, 4), dtype=np.float32)

    recon_vol[:,
    int(recon_vol.shape[1] / 2) - crop_size: int(recon_vol.shape[1] / 2) + crop_size,
    int(recon_vol.shape[2] / 2) - crop_size: int(recon_vol.shape[2] / 2) + crop_size, :] = vol

    return recon_vol


def calc_mean_std(feat, eps=1e-5):
    """
    Calculate channel-wise mean and standard deviation for the input features and preserve the dimensions
    Args:
        feat: the latent feature of shape [B, C, H, W]
        eps: a small value to prevent calculation error of variance

    Returns:
    Channel-wise mean and standard deviation of the input features
    """
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std


def adaptive_instance_normalization_with_noise(content_feat, style_feat):
    """
    Implementation of AdaIN of style transfer
    Args:
        content_feat: the content features of shape [B, C, H, W]
        style_feat: the style features of shape [B, C, H, W]

    Returns:
    The re-normalized features
    """
    size = content_feat.size()
    C = size[1]
    N = style_feat.size()[0]
    style_mean = style_feat[:, :512].view(N, C, 1, 1)
    style_std = style_feat[:, 512:].view(N, C, 1, 1)
    content_mean, content_std = calc_mean_std(content_feat)

    normalized_feat = (content_feat - content_mean.expand(
        size)) / content_std.expand(size)
    return normalized_feat * style_std.expand(size) + style_mean.expand(size)


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm2d') != -1:
        init.uniform(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)


def calc_feat_mean_std(input, eps=1e-5):
    """
    Calculate channel-wise mean and standard deviation for the input features but reduce the dimensions
    Args:
        input: the latent feature of shape [B, C, H, W]
        eps: a small value to prevent calculation error of variance

    Returns:
    Channel-wise mean and standard deviation of the input features
    """
    size = input.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = input.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C)
    feat_mean = input.view(N, C, -1).mean(dim=2).view(N, C)
    return torch.cat([feat_mean, feat_std], dim=1)


def get_checkpoints(apdx, n_epochs, mode='min', save_every_epochs=0, decoder_best_model_dir=None, decoder_model_dir=None):
    """
    Generate model checkpoints for the models in RAIN
    Args:
        decoder_model_dir: the directory to the weights of the decoder
        decoder_best_model_dir: the directory to the best weights of the decoder
        apdx: the identifier (also appendix for the files) for each weight
        n_epochs: number of epochs
        mode:
        save_every_epochs:

    Returns:

    """
    decoder_best_model_dir = 'weights/best_decoder.{}.pt'.format(apdx) if decoder_best_model_dir is None else decoder_best_model_dir
    decoder_model_dir = 'weights/decoder.{}.pt'.format(apdx) if decoder_model_dir is None else decoder_model_dir
    decoder_checkpoint = ModelCheckPointCallback(n_epochs=n_epochs, save_best=True,
                                                 mode=mode,
                                                 best_model_dir=decoder_best_model_dir,
                                                 save_last_model=True,
                                                 model_name=decoder_model_dir,
                                                 entire_model=False,
                                                 save_every_epochs=save_every_epochs)

    fc_encoder_best_dir = 'weights/best_fc_encoder.{}.pt'.format(apdx)
    fc_encoder_dir = 'weights/fc_encoder.{}.pt'.format(apdx)
    fc_encoder_checkpoint = ModelCheckPointCallback(n_epochs=n_epochs, save_best=True,
                                                    mode=mode,
                                                    best_model_dir=fc_encoder_best_dir,
                                                    save_last_model=True,
                                                    model_name=fc_encoder_dir,
                                                    entire_model=False,
                                                    save_every_epochs=save_every_epochs)

    fc_decoder_best_dir = 'weights/best_fc_decoder.{}.pt'.format(apdx)
    fc_decoder_dir = 'weights/fc_decoder.{}.pt'.format(apdx)
    fc_decoder_checkpoint = ModelCheckPointCallback(n_epochs=n_epochs, save_best=True,
                                                    mode=mode,
                                                    best_model_dir=fc_decoder_best_dir,
                                                    save_last_model=True,
                                                    model_name=fc_decoder_dir,
                                                    entire_model=False,
                                                    save_every_epochs=save_every_epochs)

    return decoder_checkpoint, fc_encoder_checkpoint, fc_decoder_checkpoint
