# Few-shot-UDA
 The official code for our [BVM 2022](https://www.bvm-workshop.org/) paper "Few-shot Unsupervised Domain Adaptation for Multi-modal Cardiac Image Segmentation"

<p align="center">
<img src="https://github.com/MingxuanGu/Few-shot-UDA/blob/main/images/architecture_bfs.png" width=900>
<p>
  
## Abstract
>Unsupervised domain adaptation (UDA) methods intend to reduce the gap between source and target domains by using unlabeled target domain and labeled source domain data, however, in the medical domain, target domain data may not always be easily available, and acquiring new samples is generally time-consuming. This restricts the development of UDA methods for new domains. In this paper, we explore the potential of UDA in a more challenging while realistic scenario where only one unlabeled target patient sample is available. We call it Few-shot Unsupervised Domain adaptation (FUDA). We first generate target-style images from source images and explore diverse target styles from a single target patient with Random Adaptive Instance Normalization (RAIN). Then, a segmentation network is trained in a supervised manner with the generated target images. Our experiments demonstrate that FUDA improves the segmentation performance by 0.33 of Dice score on the target domain compared with the baseline, and it also gives 0.28 of Dice score improvement in a more rigorous one-shot setting.

## Dataset
* Download the Multi-sequence Cardiac MR Segmentation Challenge (MS-CMRSeg 2019) dataset: 
       https://zmiclab.github.io/zxh/0/mscmrseg19/
* Data structure:
  - trainA/trainAmask: bSSFP/T2 sample 6-45
  - testA/testAmask: bSSFP/T2 sample 1-5
  - trainB/trainBmask: LGE sample 6-45
  - testB/testBmask: LGE sample 1-5
* To preprocess the data, check `preprocess_data.py`. You may need to modify the file paths to run the code.
  
## Download pretrained RAIN
* Download [vgg encoder](https://drive.google.com/file/d/1gi252ul37wIVXKbztrPV-KImLyOTeFHZ/view?usp=sharing), [fc encoder](https://drive.google.com/file/d/1PXHne-CrHLuev8PHGPR_EXHtHfaTirl1/view?usp=sharing), [fc decoder](https://drive.google.com/file/d/1QVaKtqUnbklE0L123TrI4Fzn4d-iUf-S/view?usp=sharing), [decoder](https://drive.google.com/file/d/18i02LQxuoOUi-svJ5iDHhl1wah6FbZSl/view?usp=sharing) and put them under ```pretrained/```.
 ## Installation
```
git clone https://github.com/MingxuanGu/Few-shot-UDA/
cd Few-shot-UDA
```
 ## Training
To train RAIN module:
```
Example: python3 train_RAIN.py --style_weight 5 --content_weight 5 --latent_weight 1 --recons_weight 5 --vgg ../ASM_SV/pretrained/vgg_normalised.pth --augmentation
```  
To train the DR-UNet
```
Example: python3 train_FUDA.py --backbone dr_unet --mode fewshot --jac --learning-rate 3.5e-4 --power 0.5 --eps_iters 3 --learning-rate-s 120 --num-steps 100 --num-steps-stop 100 --warmup-steps 0 --vgg_decoder pretrained/best_decoder.bssfp2t2.lr0.0001.sw5.0.cw5.0.lw1.0.rw5.0.aug.e200.Scr7.691.pt --style_encoder pretrained/best_fc_encoder.bssfp2t2.lr0.0001.sw5.0.cw5.0.lw1.0.rw5.0.aug.e200.Scr7.691.pt --style_decoder pretrained/best_fc_decoder.bssfp2t2.lr0.0001.sw5.0.cw5.0.lw1.0.rw5.0.aug.e200.Scr7.691.pt --restore_from pretrained/best_DR_UNet.fewshot.eps2.lrs20.0.pat_10_lge.e70.Scr0.58.pt
```
 ## Download Our Pretrained Weights
 We also provide the pretrained DR_Unet for direct evaluation.
 * DR_UNet after warm-up [DR_UNet after warm-up under oneshot constrain](https://drive.google.com/file/d/1Yu2t4bqL0LWXqswXUrfna7VkhPtsZYyr/view?usp=sharing), [DR_UNet after warm-up under fewshot constrain](https://drive.google.com/file/d/1cQ24pl0DhgyW7mgQX30WAu_rU2Qqe83j/view?usp=sharing)
 
* DR_UNet [trained DR_UNet under oneshot constrain](https://drive.google.com/file/d/1K_w2nW_bOnh0qJxgsMVMkch6bF933sdy/view?usp=sharing)(Dice 0.58), [trained DR_UNet under fewshot constrain](https://drive.google.com/file/d/1pyT3-xIZVHw_ZeMlqAdgxTGKLzJgMPbL/view?usp=sharing)(Dice 0.63)
 
 ## Evaluation
 ```
 Example: python3 evaluator.py --restore_from "weights/best_DR_UNet.fewshot.eps2.lrs40.0.pat_10_lge.e77.Scr0.625.pt"
 ```
 ## Results
<p align="center">
<img src="https://github.com/MingxuanGu/Few-shot-UDA/blob/main/images/3260-fewshot.png" width=900>
<p>
 
## Acknowledge
Our project is based on code from: [Adversarial Style Mining for One-Shot Unsupervised Domain Adaptation (ASM)](https://github.com/RoyalVane/ASM).
 
## Citation
Please consider citing the following paper in your publications if they help your research.
```
 @inproceedings{gu2022bvm,
  title={Few-shot Unsupervised Domain Adaptation for Multi-modal Cardiac Image Segmentation},
  author={M. {Gu} and S. {Vesal} and R. {Kosti} and A. {Maier}},
  booktitle={Bildverarbeitung für die Medizin},  
  year={2022}
}
```
