import argparse
import os
import torch.optim as optim
from engine_cpc import *
from utils.check_point import CheckPoint
from utils.history import History
from utils.utilities import get_logger
import numpy as np
import random
from torch.utils.data import DataLoader
from models import ARCH_REGISTRY
from data import DATA_REGISTRY
from data.data_transformer import RandomCropWav, Compose, TimeStretch, FakePitchShift
from layers.base import LabelSmoothingLoss, CPCBCEWithLogitsLabelSmoothingLoss
import pprint
import glob
import yaml


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def get_next_run(exp_dir):
    """
    get run id by looking at current exp_dir
    """
    next_run = 0
    files = glob.glob(os.path.join(exp_dir, "*.log"))
    for file in files:
        filename = os.path.basename(file)
        run = filename.split('.')[0]
        id = int(run[3:])
        if id >= next_run:
            next_run = id

    next_run += 1
    return 'Run{:03d}'.format(next_run)


def run(args):

    set_seed(args.seed)
    exp_dir = '{}/ckpt/{}/'.format(ROOT_DIR, args.exp)
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)

    if args.ckpt_prefix == 'auto':
        args.ckpt_prefix = get_next_run(exp_dir)

    # setup logging info
    log_file = '{}/{}.log'.format(exp_dir, args.ckpt_prefix)
    logger = get_logger(log_file)
    logger.info('\n'+pprint.pformat(vars(args)))

    # set up cuda device
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device)
    device = torch.device('cuda')

    dataset_cls = DATA_REGISTRY.get(args.dataset)
    time_stretch_args = eval(args.time_stretch_args)
    transform = Compose([
        FakePitchShift(target_sr=args.sr, pitch_shift_steps=eval(args.pitch_shift_steps)),
        RandomCropWav(target_sr=args.sr, crop_seconds=args.crop_seconds) if args.crop_seconds > 0 else None,
        TimeStretch(target_sr=args.sr, stretch_args=time_stretch_args) if time_stretch_args[0] > 0 else None
    ])
    train_set = dataset_cls(fold=args.fold,
                            split='train',
                            target_sr=args.sr,
                            transform=transform)
    train_loader = DataLoader(dataset=train_set,
                              batch_size=args.batch_size, drop_last=True, shuffle=True, num_workers=4)
    val_loader = DataLoader(dataset=dataset_cls(fold=args.fold, split='valid', target_sr=args.sr),
                            batch_size=32, drop_last=False, shuffle=False, num_workers=4)

    model_cls = ARCH_REGISTRY.get(args.net)
    model = model_cls(**vars(args)).to(device)

    logger.info(model)

    for param in model.feat.parameters():
        param.requires_grad = False

    if args.optimizer == 'sgd':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),
                              lr=args.init_lr, momentum=0.9, nesterov=True)
    elif args.optimizer == 'adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                               lr=args.init_lr, weight_decay=args.l2)
    elif args.optimizer == 'adamw':
        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                                lr=args.init_lr, weight_decay=args.l2)

    lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.init_lr,
                                                       steps_per_epoch=len(train_loader),
                                                       epochs=args.run_epochs,
                                                       pct_start=0.08,
                                                       cycle_momentum=False)

    train_hist, val_hist = History(name='train'), History(name='val')

    # checkpoint after new History, order matters
    ckpter = CheckPoint(model=model, optimizer=optimizer, path='{}/ckpt/{}'.format(ROOT_DIR, args.exp),
                        prefix=args.ckpt_prefix, interval=1, save_num=1, fake_save=False)

    criterion = LabelSmoothingLoss(smoothing=args.label_smoothing)
    if args.cpc_loss == 'info_nce':
        criterion_cpc = LabelSmoothingLoss(smoothing=args.cpc_label_smoothing)
    elif args.cpc_loss == 'bce':
        criterion_cpc = CPCBCEWithLogitsLabelSmoothingLoss(smoothing=args.cpc_label_smoothing,
                                                           balanced=False)
    elif args.cpc_loss == 'balanced_bce':
        criterion_cpc = CPCBCEWithLogitsLabelSmoothingLoss(smoothing=args.cpc_label_smoothing,
                                                           balanced=True)
    else:
        raise ValueError('cpc_loss not supported.')

    from torch.utils.tensorboard import SummaryWriter
    train_writer = SummaryWriter('{}/ckpt/{}/{}/{}'.format(ROOT_DIR, args.exp, args.ckpt_prefix, 'train'))
    valid_writer = SummaryWriter('{}/ckpt/{}/{}/{}'.format(ROOT_DIR, args.exp, args.ckpt_prefix, 'valid'))

    train_writer.add_text(tag="args", text_string=str(args))

    for epoch in range(1, args.run_epochs+1):
        train_cls_log, train_cpc_log = train_model_cpc(train_loader, model, optimizer, criterion, criterion_cpc,
                                 device, lr_scheduler, train_writer, epoch, cpc_warm_up=args.cpc_warm_up)
        train_hist.add(logs=train_cls_log, epoch=epoch)
        train_hist.add_cpc(logs=train_cpc_log, epoch=epoch)

        valid_cls_log, valid_cpc_log = eval_model_cpc(val_loader, model, criterion, criterion_cpc,
                                device, valid_writer, epoch, cpc_warm_up=args.cpc_warm_up)
        val_hist.add(logs=valid_cls_log, epoch=epoch)
        val_hist.add_cpc(logs=valid_cpc_log, epoch=epoch)

        train_writer.add_scalar("loss", train_hist.recent['loss'], epoch)
        train_writer.add_scalar("acc", train_hist.recent['acc'], epoch)
        valid_writer.add_scalar("loss", val_hist.recent['loss'], epoch)
        valid_writer.add_scalar("acc", val_hist.recent['acc'], epoch)
        train_writer.add_scalar("lr", get_lr(optimizer), epoch)

        # plotting
        if args.plot:
            train_hist.clc_plot()
            val_hist.plot()

        # logging
        logger.info("Epoch{:04d},{:6},{}".format(epoch, train_hist.name, str(train_hist.recent)))
        logger.info("Epoch{:04d},{:6},{}".format(epoch, val_hist.name, str(val_hist.recent)))

        ckpter.check_on(epoch=epoch, monitor='acc', loss_acc=val_hist.recent)

    # explicitly save last
    ckpter.save(epoch=args.run_epochs-1, monitor='acc', loss_acc=val_hist.recent)
    train_writer.close()
    valid_writer.close()


def get_cfg(cfg_file):
    configs_dict = yaml.full_load(open(cfg_file, 'r'))
    cfg = dict()
    for k, v in configs_dict.items():
        cfg[k] = v[0]
    return cfg


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", default="config/esc_conv1d_hcpc_folds_config.yaml", type=str)
    args = parser.parse_args()
    cfg = get_cfg(args.cfg)
    run(argparse.Namespace(**cfg))
