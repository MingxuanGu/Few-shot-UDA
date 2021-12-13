from datetime import datetime
import os
import pandas as pd
import numpy as np
import torch as t
from utils.utils import load_nii, read_img, resize_volume, keep_largest_connected_components, crop_volume, \
    reconstruct_volume
from utils.timer import timeit
from metric import metrics


class Evaluator:
    """
        Evaluate the performance of a segmentation model with the raw data of bSSFP and LGE
    """
    def __init__(self, file_path='../raw_data/', class_name=('myo', 'lv', 'rv')):
        """
        Parameters
        ----------
        file_path: file path to the raw data
        class_name:
        """
        self.class_name = class_name
        self._file_path = file_path

    def evaluate_single_dataset(self, seg_model, model_name='best_model', modality='lge', phase='test', ifhd=True, ifasd=True,
                 save=False, weight_dir=None, bs=16, toprint=True, lge_train_test_split=None, cal_unctnty=False, watch_pat=None):
        """
        Function to compute the metrics for a single modality of a single dataset.
        Parameters
        ----------
        seg_model: t.nn.Module
        the segmentation module.
        model_name: str
        the model name to be saved.
        modality: str
        choose from "bssfp" and "lge".
        phase: str
        choose from "train", "valid" and "test".
        ifhd: bool
        whether to calculate HD.
        ifasd: bool
        whether to calculate ASD.
        save: bool
        whether to save the resuls as csv file.
        weight_dir: str
        specify the directory to the weight if load weight.
        bs: int
        the batch size for prediction (only for memory saving).
        toprint: bool
        whether to print out the results.
        (following are not used for FUDA)
        lge_train_test_split: int
        specify from where the training data should be splitted into training and testing data.
        cal_unctnty: bool
        whether to calculate and print out the highest uncertainty (entropy) of the prediction.
        watch_pat: int
        specify the pat_id that should be printed out its uncertainty.

        Returns a dictionary of metrics {dc: [], hd: [], asd: []}.
        -------

        """
        uncertainty_list, uncertainty_slice_list = [], []
        seg_model.eval()
        if save:
            csv_path = 'evaluation_of_models_on_{}_for_{}_{}.csv'.format(modality, phase, datetime.now().date())
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path)
            else:
                data = {'DC': [], 'HD': [], 'ASD': [], 'cat': [], 'model': [], 'pad_id': []}
                df = pd.DataFrame(data)
        if weight_dir is not None:
            try:
                seg_model.load_state_dict(t.load(weight_dir)['model_state_dict'])
            except:
                seg_model.load_state_dict(t.load(weight_dir))
            print("model loaded")
        if modality == 'lge':
            folder = 'LGE'
        elif modality == 'bssfp':
            folder = 'C0'
        else:
            raise ValueError('modality can only be \'bssfp\' or \'lge\'')
        endo_dc, myo_dc, rv_dc = [], [], []
        endo_hd, myo_hd, rv_hd = [], [], []
        endo_asd, myo_asd, rv_asd, = [], [], []
        if phase == 'valid':
            start_idx = 1
            end_idx = 6
        elif phase == 'test':
            start_idx = 6 if lge_train_test_split is None else lge_train_test_split
            end_idx = 46
        else:
            start_idx = 6
            end_idx = 46 if lge_train_test_split is None else lge_train_test_split
        for pat_id in range(start_idx, end_idx):
            # if pat_id % 20 == 0:
            #     print("Evaluating patient {}".format(pat_id))
            # test_path = sorted(glob("../input/raw_data/dataset/patient{}_LGE.nii.gz".format(pat_id)))
            mask_path = os.path.join(self._file_path, 'labels/patient{}_{}_manual.nii.gz'.format(pat_id, folder))

            nimg, affine, header = load_nii(mask_path)
            vol_resize = read_img(pat_id, nimg.shape[2], modality=modality, file_path='../data/mscmrseg')
            vol_resize = crop_volume(vol_resize, crop_size=112)
            x_batch = np.array(vol_resize, np.float32) / 255.
            x_batch = np.moveaxis(x_batch, -1, 1)
            pred = []
            # temp = []
            for i in range(0, len(x_batch), bs):
                index = np.arange(i, min(i + bs, len(x_batch)))
                imgs = x_batch[index]
                pred1, pred_norm = seg_model(t.tensor(imgs).cuda())
                # uncertainty = F.softmax(pred1, dim=1).cpu().detach().numpy()
                # temp.append(uncertainty)
                pred1 = pred1.cpu().detach().numpy()
                pred.append(pred1)
            # temp = np.clip(np.concatenate(temp, axis=0), 1e-6, 1-1e-6)
            # temp = np.mean(-temp * np.log(temp), axis=(1,2,3))
            # uncertainty_slice_list.append(temp)
            # uncertainty_list.append(np.mean(temp))
            pred = np.concatenate(pred, axis=0)
            pred = np.moveaxis(pred, 1, 3)
            pred = reconstruct_volume(pred, crop_size=112)
            pred_resize = []
            for i in range(0, 4):
                pred_resize.append(resize_volume(pred[:, :, :, i], w=nimg.shape[0], h=nimg.shape[1]))
            pred = np.stack(np.array(pred_resize), axis=3)
            pred = np.argmax(pred, axis=3)

            masks = nimg.T
            masks = np.where(masks == 200, 1, masks)
            masks = np.where(masks == 500, 2, masks)
            masks = np.where(masks == 600, 3, masks)
            pred = keep_largest_connected_components(pred)
            pred = np.array(pred).astype(np.uint16)
            res = metrics(masks, pred, apply_hd=ifhd, apply_asd=ifasd, pat_id=pat_id, modality=modality,
                          class_name=self.class_name)
            if save:
                df2 = pd.DataFrame([[res['lv'][0], res['lv'][1], res['lv'][2], 'lv', model_name, pat_id],
                                    [res['rv'][0], res['rv'][1], res['rv'][2], 'rv', model_name, pat_id],
                                    [res['myo'][0], res['myo'][1], res['myo'][2], 'myo', model_name, pat_id]],
                                   columns=['DC', 'HD', 'ASD', 'cat', 'model', 'pad_id'])
                df = df.append(df2, ignore_index=True)
            # endo, rv, myo
            endo_dc.append(res['lv'][0])
            rv_dc.append(res['rv'][0])
            myo_dc.append(res['myo'][0])
            if res['lv'][1] != -1:
                endo_hd.append(res['lv'][1])
            if res['rv'][1] != -1:
                rv_hd.append(res['rv'][1])
            if res['myo'][1] != -1:
                myo_hd.append(res['myo'][1])
            if res['lv'][2] != -1:
                endo_asd.append(res['myo'][2])
            if res['rv'][2] != -1:
                rv_asd.append(res['rv'][2])
            if res['myo'][2] != -1:
                myo_asd.append(res['myo'][2])
        if cal_unctnty:
            pat_highest_ucty = np.argmax(uncertainty_list) + start_idx
            print("The pat id with the highest uncertainty: {}".format(pat_highest_ucty))
            print("The slice with the highest uncertainty in the pat {}: {}".format(pat_highest_ucty, np.argmax(uncertainty_slice_list[np.argmax(uncertainty_list)])))
            print("The pat id with the lowest uncertainty: {}".format(np.argmin(uncertainty_list) + start_idx))
            if watch_pat:
                print("The slice with the highest uncertainty in the pat {}: {}".format(watch_pat, np.argmax(
                    uncertainty_slice_list[watch_pat - start_idx])))
                print("Uncertainty of the slices of pat {}: {}".format(watch_pat, uncertainty_slice_list[watch_pat - start_idx]))
            print("Uncertainty list: {}".format(np.round(uncertainty_list, 5)))
            print("The patient with the highest DC: {}".format(np.argmax(endo_dc) + start_idx))
            print("The patient with the lowest DC: {}".format(np.argmin(endo_dc) + start_idx))
            print("DC list: {}".format(np.round(endo_dc, 3)))
        if save:
            df.to_csv(csv_path, index=False)
        mean_endo_dc = np.around(np.mean(np.array(endo_dc)), 3)
        mean_rv_dc = np.around(np.mean(np.array(rv_dc)), 3)
        mean_myo_dc = np.around(np.mean(np.array(myo_dc)), 3)
        std_endo_dc = np.around(np.std(np.array(endo_dc)), 3)
        std_rv_dc = np.around(np.std(np.array(rv_dc)), 3)
        std_myo_dc = np.around(np.std(np.array(myo_dc)), 3)
        if toprint:
            print("Modality: {}, Phase: {}".format(modality, phase))
            print("Ave endo DC: {}, {}, Ave rv DC: {}, {}, Ave myo DC: {}, {}".format(mean_endo_dc, std_endo_dc, mean_rv_dc,
                                                                                      std_rv_dc, mean_myo_dc, std_myo_dc))
            print("Ave Dice: {:.3f}, {:.3f}".format((mean_endo_dc + mean_rv_dc + mean_myo_dc) / 3.,
                                                    (std_endo_dc + std_rv_dc + std_myo_dc) / 3.))

        if ifhd:
            mean_endo_hd = np.around(np.mean(np.array(endo_hd)), 3)
            mean_rv_hd = np.around(np.mean(np.array(rv_hd)), 3)
            mean_myo_hd = np.around(np.mean(np.array(myo_hd)), 3)
            std_endo_hd = np.around(np.std(np.array(endo_hd)), 3)
            std_rv_hd = np.around(np.std(np.array(rv_hd)), 3)
            std_myo_hd = np.around(np.std(np.array(myo_hd)), 3)
            if toprint:
                print("Ave endo HD: {}, {}, Ave rv HD: {}, {}, Ave myo HD: {}, {}".format(mean_endo_hd, std_endo_hd,
                                                                                          mean_rv_hd, std_rv_hd,
                                                                                          mean_myo_hd, std_myo_hd))
                print("Ave HD: {:.3f}, {:.3f}".format((mean_endo_hd + mean_rv_hd + mean_myo_hd) / 3.,
                                                      (std_endo_hd + std_rv_hd + std_myo_hd) / 3.))
        else:
            mean_myo_hd, std_myo_hd, mean_endo_hd, std_endo_hd, mean_rv_hd, std_rv_hd = 0, 0, 0, 0, 0, 0
        if ifasd:
            mean_endo_asd = np.around(np.mean(np.array(endo_asd)), 3)
            mean_rv_asd = np.around(np.mean(np.array(rv_asd)), 3)
            mean_myo_asd = np.around(np.mean(np.array(myo_asd)), 3)
            std_endo_asd = np.around(np.std(np.array(endo_asd)), 3)
            std_rv_asd = np.around(np.std(np.array(rv_asd)), 3)
            std_myo_asd = np.around(np.std(np.array(myo_asd)), 3)
            if toprint:
                print("Ave endo ASD: {}, {}, Ave rv ASD: {}, {}, Ave myo ASD: {}, {}".format(mean_endo_asd, std_endo_asd,
                                                                                             mean_rv_asd, std_rv_asd,
                                                                                             mean_myo_asd, std_myo_asd))
                print("Ave ASD: {:.3f}, {:.3f}".format((mean_endo_asd + mean_rv_asd + mean_myo_asd) / 3.,
                                                       (std_endo_asd + std_rv_asd + std_myo_asd) / 3.))
        else:
            mean_myo_asd, std_myo_asd, mean_endo_asd, std_endo_asd, mean_rv_asd, std_rv_asd = 0, 0, 0, 0, 0, 0

        if toprint:
            print(
                'DC: {}, {}, {}, {}, {}, {}'.format(mean_myo_dc, std_myo_dc, mean_endo_dc, std_endo_dc, mean_rv_dc, std_rv_dc))
            if ifhd:
                print('HD: {}, {}, {}, {}, {}, {}'.format(mean_myo_hd, std_myo_hd, mean_endo_hd, std_endo_hd, mean_rv_hd,
                                                      std_rv_hd))
            if ifasd:
                print('ASD: {}, {}, {}, {}, {}, {}'.format(mean_myo_asd, std_myo_asd, mean_endo_asd, std_endo_asd, mean_rv_asd,
                                                  std_rv_asd))
        return {'dc': [mean_myo_dc, std_myo_dc, mean_endo_dc, std_endo_dc, mean_rv_dc, std_rv_dc],
                'hd': [mean_myo_hd, std_myo_hd, mean_endo_hd, std_endo_hd, mean_rv_hd, std_rv_hd],
                'asd': [mean_myo_asd, std_myo_asd, mean_endo_asd, std_endo_asd, mean_rv_asd, std_rv_asd]}

    @timeit
    def evaluate(self, seg_model, ifhd=True, ifasd=True, weight_dir=None, bs=16, lge_train_test_split=None):
        bssfp_train = self.evaluate_single_dataset(seg_model=seg_model, modality='bssfp', phase='train', ifhd=ifhd, ifasd=ifasd, save=False, weight_dir=weight_dir, bs=bs, toprint=False)
        bssfp_val = self.evaluate_single_dataset(seg_model=seg_model, modality='bssfp', phase='valid', ifhd=ifhd, ifasd=ifasd, save=False, weight_dir=weight_dir, bs=bs, toprint=False)
        lge_val = self.evaluate_single_dataset(seg_model=seg_model, modality='lge', phase='valid', ifhd=ifhd, ifasd=ifasd, save=False, weight_dir=weight_dir, bs=bs, toprint=False)
        lge_test = self.evaluate_single_dataset(seg_model=seg_model, modality='lge', phase='test', ifhd=ifhd, ifasd=ifasd, save=False, weight_dir=weight_dir, bs=bs, toprint=False,
                                                lge_train_test_split=lge_train_test_split)

        return bssfp_train, bssfp_val, lge_val, lge_test


if __name__ == '__main__':
    import argparse
    from model.DRUNet import Segmentation_model as DR_UNet
    from torch.cuda import get_device_name
    print("Device name: {}".format(get_device_name(0)))
    parser = argparse.ArgumentParser(description="Evaluation")
    parser.add_argument("--restore_from", type=str, default='weights/best_DR_UNet.fewshot.lr0.00035.eps3.LSeg.lrs120.0.pat_10_lge.e40.Scr0.67.pt', help="Where restore model parameters from.")
    parser.add_argument("--batch_size", type=int, default=16, help="Number of images sent to the network in one step.")
    parser.add_argument("--file_path", type=str, default='../data/mscmrseg/raw_data')
    args = parser.parse_args()
    evaluator = Evaluator(file_path=args.file_path)
    segmentor = DR_UNet(n_class=4)
    evaluator.evaluate_single_dataset(segmentor, model_name='best_model', modality='lge', phase='test', ifhd=True,
                                      ifasd=True, save=False, weight_dir=args.restore_from, bs=args.batch_size,
                                      toprint=True, lge_train_test_split=None, cal_unctnty=False, watch_pat=None)