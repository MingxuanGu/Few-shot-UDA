import argparse
from datetime import datetime, timedelta
import torch
from torch import nn
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import model.RAIN as net
from dataset.data_generator_RAIN import DataGenerator
from utils.utils import get_checkpoints
from utils.timer import timeit

print("Device name: {}".format(torch.cuda.get_device_name(0)))
start_time = datetime.now()
max_duration = 24 * 3600 - 5 * 60
print("torch version: {}".format(torch.__version__))
print("device count: {}".format(torch.cuda.device_count()))
print('device name: {}'.format(torch.cuda.get_device_name(0)))

cudnn.benchmark = True

torch.autograd.set_detect_anomaly(True)


def adjust_learning_rate(optimizer, iteration_count):
    """Imitating the original implementation"""
    lr = args.lr / (1.0 + args.lr_decay * iteration_count)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


parser = argparse.ArgumentParser()
# Basic options
parser.add_argument('--content', type=str, default='bssfp',
                    help='the modality for content')  # bssfp, t2 or lge
parser.add_argument('--style', type=str, default='t2',
                    help='the modality for content')

parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--lr_decay', type=float, default=5e-5)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--ns', type=int, default=700)
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--augmentation', action='store_true')
parser.add_argument('--lw', action='store_true')
parser.add_argument('--crop', type=int, default=224)
parser.add_argument('--vgg', help='the path to the directory of the weight', type=str,
                    default='pretrained/vgg_normalised.pth')
parser.add_argument('--style_weight', type=float, default=1.0)
parser.add_argument('--content_weight', type=float, default=1.0)
parser.add_argument('--latent_weight', type=float, default=1.0)
parser.add_argument('--recons_weight', type=float, default=5.0)
parser.add_argument('--save_every_epochs', type=int, default=0)
args = parser.parse_args()
print('weight_dir: {}'.format(args.vgg))


def get_appendix():
    appendix = args.content + "2" + args.style + '.lr{}'.format(args.lr) + \
               '.sw{}'.format(args.style_weight) + '.cw{}'.format(args.content_weight) + \
               '.lw{}'.format(args.latent_weight) + '.rw{}'.format(args.recons_weight)
    if args.augmentation:
        appendix += '.aug'
    return appendix


appendix = get_appendix()
print(appendix)
device = torch.device('cuda')


@timeit
def main():
    decoder = net.decoder
    vgg = net.encoder
    fc_encoder = net.fc_encoder
    fc_decoder = net.fc_decoder

    vgg.load_state_dict(torch.load(args.vgg))
    vgg = nn.Sequential(*list(vgg.children())[:31])

    start_epoch = 0
    if args.lw:
        decoder_model_dir = 'weights/decoder.{}.pt'.format(appendix)
        fc_encoder_dir = 'weights/fc_encoder.{}.pt'.format(appendix)
        fc_decoder_dir = 'weights/fc_decoder.{}.pt'.format(appendix)
        decoder.load_state_dict(torch.load(decoder_model_dir)['model_state_dict'])
        fc_encoder.load_state_dict(torch.load(fc_encoder_dir)['model_state_dict'])
        fc_decoder.load_state_dict(torch.load(fc_decoder_dir)['model_state_dict'])
        start_epoch = torch.load(decoder_model_dir)['epoch']
    print('start epoch: {}'.format(start_epoch))
    network = net.Net(vgg, decoder, fc_encoder, fc_decoder)
    network.train()
    network.to(device)

    decoder_checkpoint, fc_encoder_checkpoint, fc_decoder_checkpoint = get_checkpoints(apdx=appendix,
                                                                                       n_epochs=args.epochs,
                                                                                       save_every_epochs=args.save_every_epochs)

    content_iter = iter(
        DataGenerator(modality=args.content, crop_size=args.crop, n_samples=args.ns, batch_size=args.batch_size,
                      augmentation=args.augmentation))
    style_iter = iter(
        DataGenerator(modality=args.style, crop_size=args.crop, n_samples=args.ns, batch_size=args.batch_size,
                      augmentation=args.augmentation))

    optimizer = torch.optim.Adam(network.decoder.parameters(), lr=args.lr)
    optimizer_fc_encoder = torch.optim.Adam(network.fc_encoder.parameters(), lr=args.lr)
    optimizer_fc_decoder = torch.optim.Adam(network.fc_decoder.parameters(), lr=args.lr)

    loss_c_mean_list, loss_s_mean_list, loss_l_mean_list, loss_r_mean_list = [], [], [], []
    decoder_lr = []
    fc_encoder_lr = []
    fc_decoder_lr = []
    max_epoch_time = 0
    for i in tqdm(range(start_epoch, args.epochs)):
        epoch_start = datetime.now()
        loss_c_list, loss_s_list, loss_l_list, loss_r_list = [], [], [], []

        adjust_learning_rate(optimizer, iteration_count=i)
        adjust_learning_rate(optimizer_fc_encoder, iteration_count=i)
        adjust_learning_rate(optimizer_fc_decoder, iteration_count=i)

        decoder_lr.append(optimizer.param_groups[0]['lr'])
        fc_encoder_lr.append(optimizer_fc_encoder.param_groups[0]['lr'])
        fc_decoder_lr.append(optimizer_fc_decoder.param_groups[0]['lr'])

        for (content_images, _), (style_images, _) in zip(content_iter, style_iter):
            for param in network.fc_encoder.parameters():
                param.requires_grad = True
            for param in network.fc_decoder.parameters():
                param.requires_grad = True
            loss_c, loss_s, loss_l, loss_r = network.forward(torch.from_numpy(content_images).cuda(),
                                                             torch.from_numpy(style_images).cuda())

            # collect losses
            loss_c_list.append(loss_c.item())
            loss_s_list.append(loss_s.item())
            loss_l_list.append(loss_l.item())
            loss_r_list.append(loss_r.item())

            loss_l = args.latent_weight * loss_l
            loss_r = args.recons_weight * loss_r
            loss_fc = loss_l + loss_r
            optimizer_fc_encoder.zero_grad()
            optimizer_fc_decoder.zero_grad()
            loss_fc.backward(retain_graph=True)

            for param in network.fc_encoder.parameters():
                param.requires_grad = False
            for param in network.fc_decoder.parameters():
                param.requires_grad = False

            loss_c = args.content_weight * loss_c
            loss_s = args.style_weight * loss_s
            optimizer.zero_grad()
            loss_de = loss_c + loss_s
            loss_de.backward()
            optimizer.step()
            optimizer_fc_encoder.step()
            optimizer_fc_decoder.step()
            print("{}, {}, {}, {}, {}".format("0:>12".format('epoch: {}'.format(i)), loss_c.item(), loss_s.item(),
                                              loss_l.item(), loss_r.item()))

        mean_loss_c = sum(loss_c_list) / len(loss_c_list)
        mean_loss_s = sum(loss_s_list) / len(loss_s_list)
        mean_loss_l = sum(loss_l_list) / len(loss_l_list)
        mean_loss_r = sum(loss_r_list) / len(loss_r_list)
        loss_c_mean_list.append(mean_loss_c)
        loss_s_mean_list.append(mean_loss_s)
        loss_l_mean_list.append(mean_loss_l)
        loss_r_mean_list.append(mean_loss_r)
        if (datetime.now() - start_time).seconds > max_duration - max_epoch_time:
            i = args.epochs - 1

        monitor = args.latent_weight * mean_loss_l + args.recons_weight * mean_loss_r + \
                  args.content_weight * mean_loss_c + args.style_weight * mean_loss_s
        monitor = round(monitor, 3)

        decoder_checkpoint.step(monitor=monitor, model=network.decoder, epoch=i + 1)
        fc_encoder_checkpoint.step(monitor=monitor, model=network.fc_encoder, epoch=i + 1)
        fc_decoder_checkpoint.step(monitor=monitor, model=network.fc_decoder, epoch=i + 1)

        if (datetime.now() - start_time).seconds > max_duration - max_epoch_time:
            print("training time elapsed: {}".format(datetime.now() - start_time))
            print("max_epoch_time: {}".format(timedelta(seconds=max_epoch_time)))
            break
        max_epoch_time = max((datetime.now() - epoch_start).seconds, max_epoch_time)

    writer = SummaryWriter(comment=appendix + 'Scr{}'.format(decoder_checkpoint.best_result))
    print("write a training summary")
    i = 1
    for loss_c, loss_s, loss_l, loss_r, d_lr, fc_e_lr, fc_d_lr in zip(loss_c_mean_list, loss_s_mean_list,
                                                                      loss_l_mean_list, loss_r_mean_list,
                                                                      decoder_lr, fc_encoder_lr, fc_decoder_lr):
        writer.add_scalars('Loss', {'Content': loss_c, 'Style': loss_s, 'KL': loss_l, 'Reconstruct': loss_r,
                                    'Combine': args.latent_weight * loss_l + args.recons_weight * loss_r +
                                               args.content_weight * loss_c + args.style_weight * loss_s}, i)
        writer.add_scalars('Lr', {'Decoder': d_lr, 'FC_encoder': fc_e_lr, 'FC_decoder': fc_d_lr}, i)
        i += 1
    writer.close()


if __name__ == '__main__':
    main()
    print('Time elapsed: {}'.format(datetime.now() - start_time))
    print('program finish')
