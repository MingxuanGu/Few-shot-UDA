import argparse
import tqdm
import numpy as np
import glob
import os
import cv2
from pathlib import Path
from datetime import datetime
import torch
import torch.nn as nn
from torch.utils import data
from torch.autograd import Variable
import torch.optim as optim
import torch.backends.cudnn as cudnn
from model.DRUNet import Segmentation_model as DR_UNet
from model.RAIN import encoder, decoder, fc_encoder, fc_decoder, device
from utils.loss import jaccard_loss
from utils.callbacks import ModelCheckPointCallback
from utils.utils import adaptive_instance_normalization_with_noise, calc_feat_mean_std
from dataset.bSSFP_dataset import bSSFPDataSet
from dataset.LGE_dataset import LGEDataSet
from evaluator import Evaluator


MODEL = 'dr_unet'
BATCH_SIZE = 4
NUM_WORKERS = 2

MOMENTUM = 0.9
NUM_CLASSES = 4

SAVE_PRED_EVERY = 5

#Hyper Paramters
WEIGHT_DECAY = 0.0005
LEARNING_RATE = 2.5e-4
LEARNING_RATE_S = 20

POWER = 0.9
RANDOM_SEED = 1234

INPUT_SIZE_SOURCE = '224,224'
DATA_DIRECTORY = "../data/mscmrseg/"
NUM_STEPS = 40
NUM_STEPS_STOP = NUM_STEPS
WARMUP_STEPS = NUM_STEPS
EPS_ITERS = 2


def get_arguments():
    """Parse all the arguments.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="DeepLab-ResNet Network")
    parser.add_argument("--model", type=str, default=MODEL,
                        help="available options : ResNet")
    parser.add_argument("--backbone", type=str, default="deeplab",
                        help="available options: deeplab, dr-unet")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE,
                        help="Number of images sent to the network in one step.")
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS,
                        help="number of workers for multithread dataloading.")
    parser.add_argument("--data-dir", type=str, default=DATA_DIRECTORY,
                        help="Path to the directory containing the source dataset.")
    parser.add_argument("--is-training", action="store_true",
                        help="Whether to updates the running means and variances during the training.")
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE,
                        help="Base learning rate for training with polynomial decay.")
    parser.add_argument("--learning-rate-s", type=float, default=LEARNING_RATE_S,
                        help="Base learning rate for epsilon(sampling).")
    parser.add_argument("--eps_steps", type=float, default=2,
                        help="Base learning rate for epsilon(sampling).")
    parser.add_argument("--momentum", type=float, default=MOMENTUM,
                        help="Momentum component of the optimiser.")
    parser.add_argument("--not-restore-last", action="store_true",
                        help="Whether to not restore last (FC) layers.")
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--num-steps", type=int, default=NUM_STEPS,
                        help="Number of training steps.")
    parser.add_argument("--num-steps-stop", type=int, default=NUM_STEPS_STOP,
                        help="Number of training steps for early stopping.")
    parser.add_argument("--warmup-steps", type=int, default=WARMUP_STEPS,
                        help="Number of training steps for early stopping.")
    parser.add_argument("--eps_iters", type=int, default=EPS_ITERS,
                        help="Number of iterations for each epsilon.")
    parser.add_argument("--power", type=float, default=POWER,
                        help="Decay parameter to compute the learning rate.")
    parser.add_argument("--random-seed", type=int, default=RANDOM_SEED,
                        help="Random seed to have reproducible results.")
    parser.add_argument("--restore_from", type=str, default=None,
                        help="Where restore model parameters from.")
    parser.add_argument("--save-pred-every", type=int, default=SAVE_PRED_EVERY,
                        help="Save summaries and checkpoint every often.")
    parser.add_argument("--style_dir", type=str, default='style_track',
                        help="Where to save style images of the model.")
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY,
                        help="Regularisation parameter for L2-loss.")
    parser.add_argument("--gpu", type=int, default=0,
                        help="choose gpu device.")
    parser.add_argument('--vgg_encoder', type=str, default='pretrained/vgg_normalised.pth')
    parser.add_argument("--mode", help="whether train the model in 'fewshot', 'oneshot'", type=str, default='oneshot')
    parser.add_argument('--vgg_decoder', type=str, default='pretrained/decoder_iter_100000.pth')
    parser.add_argument('--style_encoder', type=str, default='pretrained/fc_encoder_iter_100000.pth')
    parser.add_argument('--style_decoder', type=str, default='pretrained/fc_decoder_iter_100000.pth')
    parser.add_argument('--fp16', action='store_true',
                        help='use float16 instead of float32, which will save about 50% memory')
    parser.add_argument('--jac', help='whether to apply jaccard loss', action='store_true')
    return parser.parse_args()


args = get_arguments()


def loss_calc(pred, label, gpu=0, jaccard=False):
    """
    This function returns cross entropy loss plus jaccard loss for semantic segmentation
    Args:
        pred: the logits of the prediction with shape [B, C, H, W]
        label: the ground truth with shape [B, H, W]
        gpu: the gpu number
        jaccard: if apply jaccard loss

    Returns:

    """
    label = Variable(label.long()).cuda(gpu)
    criterion = nn.CrossEntropyLoss().cuda(gpu)
    loss = criterion(pred, label)
    if jaccard:
        loss += jaccard_loss(true=label, logits=pred)
    return loss


def lr_poly(base_lr, iter, max_iter, power):
    return base_lr * ((1 - float(iter) / max_iter) ** (power))


def lr_warmup(base_lr, iter, warmup_iter):
    return base_lr * (float(iter) / warmup_iter)


def adjust_learning_rate(optimizer, i_iter):
    if i_iter < args.warmup_steps:
        lr = args.learning_rate - lr_warmup(args.learning_rate, i_iter, args.warmup_steps)
    else:
        lr = lr_poly(args.learning_rate, i_iter, args.num_steps, args.power)
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) > 1:
        optimizer.param_groups[1]['lr'] = lr * 10


def style_transfer(encoder, decoder, fc_encoder, fc_decoder, content, style, sampling=None):
    """
    RAIN implementation that generate images which preserve content of content images and style of style images
    Args:
        encoder: the VGG encoder
        decoder: the VGG-like decoder
        fc_encoder: the VAE encoder
        fc_decoder: the VAE decoder
        content: the content images
        style: the style images
        sampling: the epsilon sampled from a distribution generated by the VAE encoder

    Returns:
    Images which preserve content of content images and style of style images, the sampling which will be updated for
    the following iterations
    """
    with torch.no_grad():
        content_feat = encoder(content)
        style_feat = encoder(style)
    style_feat_mean_std = calc_feat_mean_std(style_feat)
    intermediate = fc_encoder(style_feat_mean_std)
    intermediate_mean = intermediate[:, :512]
    intermediate_std = intermediate[:, 512:]
    noise = torch.randn_like(intermediate_mean)
    if sampling is None:
        sampling = intermediate_mean + noise * intermediate_std #N, 512
    sampling.requires_grad = True
    style_feat_mean_std_recons = fc_decoder(sampling) #N, 1024
    feat = adaptive_instance_normalization_with_noise(content_feat, style_feat_mean_std_recons)

    return decoder(feat), sampling


def main():
    """Create the model and start the training."""
    input_size_source = 224

    cudnn.enabled = True
    # the training start from epoch 0 (will be change if the model is loaded from a under-trained weights)
    start_epoch = 0
    # Create Network
    segmentor = DR_UNet(n_class=args.num_classes)
    if args.restore_from:
        checkpoint = torch.load(args.restore_from)
        # load from under-trained weights
        segmentor.load_state_dict(checkpoint['model_state_dict'])
        # read in the epoch number
        start_epoch = checkpoint['epoch'] if 'pretrained' not in args.restore_from else start_epoch
        print("model load from state dict: {}".format(os.path.basename(args.restore_from)))

    segmentor.train()
    segmentor.cuda(args.gpu)

    cudnn.benchmark = True

    weight_root_dir = './weights/'
    if not os.path.exists(weight_root_dir):
        os.mkdir(weight_root_dir)
    apdx = "DR_UNet." + args.mode + '.lr{}'.format(args.learning_rate) + '.eps{}.LSeg'.format(args.eps_iters) + '.lrs{}'.format(args.learning_rate_s)
    if args.mode == 'fewshot':
        apdx += ".pat_10_lge"
    else:
        apdx += ".pat_10_lge_13"
    weight_dir = os.path.join(weight_root_dir, apdx + '.pt')
    best_weight_dir = os.path.join(weight_root_dir, "best_" + apdx + '.pt')
    # create the model check point
    modelcheckpoint_unet = ModelCheckPointCallback(n_epochs=args.num_steps, save_best=True,
                                                   mode="max",
                                                   best_model_dir=best_weight_dir,
                                                   save_last_model=True,
                                                   model_name=weight_dir,
                                                   entire_model=False)
    # create the evaluator
    evaluator = Evaluator(file_path='../data/mscmrseg/raw_data')

    # create VGG encoder, decoder, VAE encoder, VAE decoder
    vgg_encoder = encoder
    vgg_decoder = decoder
    style_encoder = fc_encoder
    style_decoder = fc_decoder
    # freeze RAIN
    vgg_encoder.eval()
    style_encoder.eval()
    vgg_decoder.eval()
    style_decoder.eval()

    # load pretrained weights for RAIN
    vgg_encoder.load_state_dict(torch.load(args.vgg_encoder))
    vgg_encoder = nn.Sequential(*list(vgg_encoder.children())[:31])
    try:
        vgg_decoder.load_state_dict(torch.load(args.vgg_decoder))
    except:
        vgg_decoder.load_state_dict(torch.load(args.vgg_decoder)['model_state_dict'])
        print("decoder load from state dict")
    try:
        style_encoder.load_state_dict(torch.load(args.style_encoder))
    except:
        style_encoder.load_state_dict(torch.load(args.style_encoder)['model_state_dict'])
        print("fc_decoder load from state dict")
    try:
        style_decoder.load_state_dict(torch.load(args.style_decoder))
    except:
        style_decoder.load_state_dict(torch.load(args.style_decoder)['model_state_dict'])
        print("fc_encoder load from state dict")

    vgg_encoder.to(device)
    vgg_decoder.to(device)
    style_encoder.to(device)
    style_decoder.to(device)

    for param in vgg_encoder.parameters():
        param.requires_grad = False
    # mkdir for the stylized images
    if not os.path.exists(args.style_dir):
        os.makedirs(args.style_dir)
    # dataloader for bSSFP images
    trainloader = data.DataLoader(
        bSSFPDataSet(args.data_dir, max_iters=None,
                    crop_size=input_size_source),
        batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    print("length of bSSFP training dataset: {}".format(len(trainloader)))
    # choose patient 10 as the training lge image
    pat_id = 10
    # dataloader for LGE images
    if args.mode == 'fewshot' or args.mode == "oneshot":
        print("{} Training".format(args.mode))
        targetloader = data.DataLoader(
            LGEDataSet(args.data_dir, max_iters=None,
                        crop_size=input_size_source, pat_id=10),
            batch_size=len(glob.glob(os.path.join(args.data_dir, "trainB/*_{}_*lge*.png".format(pat_id)))),
            shuffle=False, num_workers=args.num_workers, pin_memory=True)
    else:
        print("Fulldata Training")
        targetloader = data.DataLoader(
            LGEDataSet(args.data_dir, max_iters=None,
                       crop_size=input_size_source, mode=args.mode),
            batch_size=len(glob.glob(os.path.join(args.data_dir, "trainB/pat*lge*.png"))),
            shuffle=False, num_workers=args.num_workers, pin_memory=True)
    targetloader_iter = enumerate(targetloader)

    optimizer = optim.SGD(segmentor.parameters(), lr=args.learning_rate, momentum=args.momentum,
                          weight_decay=args.weight_decay)
    optimizer.zero_grad()
    loss_norm = nn.MSELoss()
    try:
        _, batch_t = next(targetloader_iter)
    except StopIteration:
        targetloader_iter = enumerate(targetloader)
        _, batch_t = next(targetloader_iter)
    images_t, tar_name = batch_t
    images_t = Variable(images_t).cuda(args.gpu)
    images_t.requires_grad = False

    if args.mode == 'oneshot':
        idx = 5
        images_t_temp = images_t[idx:idx + 1, ...]
        print("the image selected as target:", tar_name[idx])
        target_name = tar_name[idx]

    losses_seg = []
    losses_consistent = []
    lge_dice = []
    seg_lr = []

    for i_iter in tqdm.trange(start_epoch, args.num_steps):
        epoch_start_time = datetime.now()
        adjust_learning_rate(optimizer, i_iter)
        seg_lr.append(optimizer.param_groups[0]['lr'])
        to_save_pic = True
        loss_seg_list = []
        loss_consistent_list = []
        trainloader_iter = enumerate(trainloader)
        k = -1
        for _, batch_s in trainloader_iter:
            k += 1
            if args.mode == 'fewshot' or args.mode == 'fulldata':
                # for few shot learning in every iter we chose a random image out of patient slices
                idx = k % len(images_t)
                images_t_temp = images_t[idx:idx+1, ...]
                target_name = tar_name[idx]
            segmentor.train()
            sampling = None
            optimizer.zero_grad()
            images_s, labels_s, names = batch_s
            images_s = Variable(images_s).cuda(args.gpu)
            images_s.requires_grad = False
            images_s_temp = torch.clone(images_s).detach()
            time_iter = 1 if i_iter < args.warmup_steps else args.eps_iters
            for i in range(time_iter):
                images_s_style, sampling = style_transfer(vgg_encoder, vgg_decoder, style_encoder, style_decoder,
                                                          images_s_temp, images_t_temp, sampling)
                images_s_style = torch.mean(images_s_style, dim=1)
                images_s_style = torch.stack([images_s_style, images_s_style, images_s_style], dim=1)
                if to_save_pic and (i_iter + 1) % SAVE_PRED_EVERY == 0:
                    to_save_pic = False if (i + 1) == time_iter else to_save_pic
                    output = images_s_style.detach().cpu().numpy()[0]
                    output = np.clip(output, 0, 1)
                    output = output * 255.
                    output = output.astype(np.uint8)
                    output = np.moveaxis(output, 0, -1)
                    if time_iter == 1:
                        image_name = 'warmup/{}_iter{:d}_{}_2_{}.jpg'.format(args.mode, i_iter + 1,
                                                                             Path(names[0]).stem,
                                                                             Path(target_name).stem)
                    else:
                        if not os.path.exists('{}/{}'.format(args.style_dir, apdx)):
                            os.mkdir('{}/{}'.format(args.style_dir, apdx))
                        image_name = '{}/{}_iter{:d}_{}_2_{}_{}.jpg'.format(apdx, args.mode, i_iter + 1,
                                                                            Path(names[0]).stem,
                                                                            Path(target_name).stem, i + 1)

                    image_name = '{}/{}'.format(args.style_dir, image_name)
                    cv2.imwrite(image_name, output)
                    print("{} saved.".format(image_name))
                pred, pred_norm = segmentor(torch.cat([images_s_style, images_s], dim = 0))
                norm_loss = 0
                # calculate the consistency loss
                for norm_id in range(pred_norm.size()[0] // 2):
                    norm_loss += (pred_norm[norm_id] - pred_norm[norm_id + pred_norm.size()[0] // 2]).norm(p=2, dim=(1, 2))
                pred_norm = norm_loss / (pred_norm.size()[0] // 2)
                label_tensor = torch.cat([labels_s, labels_s], dim = 0)
                # calculate the segmentation loss
                loss_1 = loss_calc(pred, label_tensor, args.gpu, jaccard=args.jac)
                loss_2 = loss_norm(pred_norm, torch.zeros(pred_norm.size()).float().cuda())
                loss_seg_list.append(loss_1.item())
                loss_consistent_list.append(loss_2.item())
                # check whether need to retain graph
                retain_graph = (i_iter >= args.warmup_steps)
                if retain_graph:
                    sampling.require_grad = True
                    sampling.retain_grad()
                    samp_loss = loss_1
                    samp_loss.backward(retain_graph=retain_graph)
                    grad_data = sampling.grad.data
                    optimizer.zero_grad()
                loss = loss_1 + 2e-3 * loss_2
                loss.backward()
                if retain_graph:
                    sampling = sampling + (args.learning_rate_s/samp_loss.item()) * grad_data
                    sampling = Variable(sampling.detach(), requires_grad=True)
                optimizer.step()
        losses_seg.append(np.mean(loss_seg_list))
        losses_consistent.append(np.mean(loss_consistent_list))
        print('Epoch = {0:6d}/{1:6d}, loss_seg = {2:.4f} loss_con = {3:.4f}'.format(
                i_iter + 1, args.num_steps, losses_seg[-1], losses_consistent[-1]))
        results = evaluator.evaluate_single_dataset(seg_model=segmentor, ifhd=False, ifasd=False, modality='lge',
                                                    phase='valid', bs=10)
        lge_dice.append(np.round((results['dc'][0] + results['dc'][2] + results['dc'][4]) / 3, 3))
        modelcheckpoint_unet.step(monitor=lge_dice[-1], model=segmentor, epoch=i_iter + 1, optimizer=optimizer,
                                  tobreak=(i_iter + 1) == args.num_steps)
        print("Time elapsed for epoch {}: {}".format(i_iter, datetime.now() - epoch_start_time))

    print("Writing summary")
    from torch.utils.tensorboard import SummaryWriter
    log_dir = 'runs/{}.e{}.Scr{}'.format(apdx, modelcheckpoint_unet.epoch,
                                         np.around(modelcheckpoint_unet.best_result, 3))
    writer = SummaryWriter(log_dir=log_dir)
    i = 0
    for loss_seg, loss_con, dice, s_lr in zip(losses_seg, losses_consistent, lge_dice, seg_lr):
        writer.add_scalar('Loss/Training_seg', loss_seg, i)
        writer.add_scalar('Loss/Training_consistent', loss_con, i)
        writer.add_scalar('Dice/LGE_valid', dice, i)
        writer.add_scalar('LR/Seg_LR', s_lr, i)
        i += 1
    writer.close()
    # load the weights with the bext validation score and do the evaluation
    model_name = '{}.e{}.Scr{}{}'.format(modelcheckpoint_unet.best_model_name_base, modelcheckpoint_unet.epoch, np.around(modelcheckpoint_unet.best_result, 3), modelcheckpoint_unet.ext)
    print("the weight of the best unet model: {}".format(model_name))
    try:
        segmentor.load_state_dict(torch.load(model_name)['model_state_dict'])
        print("segmentor load from state dict")
    except:
        segmentor._unet.load_state_dict(torch.load(model_name))
    print("model loaded")
    evaluator.evaluate_single_dataset(seg_model=segmentor, modality='lge', phase='test', ifhd=True, ifasd=False,
                                      save=False, weight_dir=None, bs=8, toprint=True, lge_train_test_split=None)
    return

if __name__ == '__main__':
    main()
    print("program finished.")
