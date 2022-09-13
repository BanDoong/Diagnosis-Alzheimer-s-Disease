import os

import nibabel
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import argparse
import random

# Loss
from supcontrastive import SupConLoss

# Tensorboard
from torch.utils.tensorboard import SummaryWriter
from model_clinica import Conv5_FC3, Conv4_FC3
from torchvision import transforms
import yaml

# Dataset
from dataset import Dataset, MinMaxNormalization
# from run_finetune import cf_matrix, add_element, cal_metric
from utils import make_df, write_result
from resnet import generate_model
from torch.utils.data import sampler
from sklearn.metrics import confusion_matrix
from torchinfo import summary
from sklearn.ensemble import VotingClassifier


def to_cpu(inputs):
    return inputs.cpu()


# def cf_matrix(y_true, y_pred, num_label):
#     cf = ConfusionMatrix(num_classes=num_label)
#     # softmax = nn.Softmax()
#     # y_pred = softmax(y_pred)
#     _, y_pred = torch.max(y_pred, 1)
#     y_pred = to_cpu(y_pred)
#     y_true = to_cpu(y_true)
#     return cf(y_true, y_pred).flatten()

def cf_matrix(y_true, y_pred, num_label):
    _, y_pred = torch.max(y_pred, 1)
    y_pred = to_cpu(y_pred).tolist()
    y_true = to_cpu(y_true).tolist()
    out = confusion_matrix(y_true, y_pred, labels=[0, 1])
    return out.ravel()


def to_numpy(acc, spe, sen, f1, prec):
    acc = acc.numpy()
    spe = spe.numpy()
    sen = sen.numpy()
    f1 = f1.numpy()
    prec = prec.numpy()
    return acc, spe, sen, f1, prec


def add_element(cf, new):
    # return torch.as_tensor(list(map(add, cf, new)))
    return [x + y for x, y in zip(cf, new)]


def cal_metric(cf_list):
    TN, FP, FN, TP = cf_list
    # print(f'TN, FP, FN, TP : {cf_list}')
    accuracy = (TP + TN) / (FP + FN + TP + TN) if FP + FN + TP + TN != 0 else 0.
    # specificity
    specificity = TN / (TN + FP) if (TN + FP) != 0 else 0.
    precision = TP / (TP + FP) if (TP + FP) != 0 else 0.
    # sensitivity or recall
    sensitivity = TP / (TP + FN) if (TP + FN) != 0 else 0.
    bacc = (sensitivity + specificity) / 2
    # F1 = TP / (TP + (FN + FP) / 2) if TP + (FN + FP) / 2 != 0 else torch.as_tensor(0.)
    # ppv = TP / (TP + FP) if (TP + FP) != 0 else 0.
    npv = TN / (TN + FN) if (TN + FN) != 0 else 0.

    if (precision + sensitivity) != 0:
        F1 = (2 * precision * sensitivity) / (precision + sensitivity)
    else:
        F1 = 0.
    # accuracy, specificity, sensitivity, ppv, npv = to_numpy(accuracy, specificity, sensitivity, ppv, npv)
    # bacc = bacc.numpy()
    # F1 = F1.numpy()
    return accuracy, bacc, specificity, sensitivity, precision, npv, F1


class EarlyStopping:
    """주어진 patience 이후로 validation loss가 개선되지 않으면 학습을 조기 중지"""

    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', multigpu=False, pretrain=False,
                 f1=None):
        """
        Args:
            patience (int): validation loss가 개선된 후 기다리는 기간
                            Default: 7
            verbose (bool): True일 경우 각 validation loss의 개선 사항 메세지 출력
                            Default: False
            delta (float): 개선되었다고 인정되는 monitered quantity의 최소 변화
                            Default: 0
            path (str): checkpoint저장 경로
                            Default: 'checkpoint.pt'
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.f1_min = np.Inf
        self.delta = delta
        self.path = path
        self.pretrain = pretrain
        self.multigpu = multigpu
        self.f1 = f1

    def __call__(self, val_loss, model, f1):

        if self.pretrain:
            score = f1
        else:
            score = -val_loss
        if self.best_score is None:
            self.best_score = score
            if self.pretrain:
                self.save_checkpoint(f1=f1, model=model)
            else:
                self.save_checkpoint(val_loss=val_loss, model=model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'Before best value is {self.f1_min}')
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            if self.pretrain:
                self.save_checkpoint(f1=f1, model=model)
            else:
                self.save_checkpoint(val_loss=val_loss, model=model)
            self.counter = 0

    def save_checkpoint(self, val_loss=None, f1=None, model=None):
        '''validation loss가 감소하면 모델을 저장한다.'''
        if self.verbose:
            if self.pretrain:
                print(f'BACC Score is Increased ({self.f1_min} --> {f1}).  Saving model ...')
            else:
                print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        if self.multigpu:
            torch.save(model.module.state_dict(), self.path)
            print("==========================================")
        else:
            torch.save(model.state_dict(), self.path)
        if self.pretrain:
            self.f1_min = f1
        else:
            self.val_loss_min = val_loss


def yaml_config_hook(config_file):
    """
    Custom YAML config loader, which can include other yaml files (I like using config files
    insteaad of using argparser)
    """

    # load yaml files in the nested 'defaults' section, which include defaults for experiments
    with open(config_file) as f:
        cfg = yaml.safe_load(f)
        for d in cfg.get("defaults", []):
            config_dir, cf = d.popitem()
            cf = os.path.join(os.path.dirname(config_file), config_dir, cf + ".yaml")
            with open(cf) as f:
                l = yaml.safe_load(f)
                cfg.update(l)

    if "defaults" in cfg.keys():
        del cfg["defaults"]

    return cfg


def train(args, train_loader, model, criterion, optimizer, scheduler):
    loss_epoch = []
    cf_list = [0] * (args.num_label ** 2)

    for step, batch_data in enumerate(train_loader):
        if args.roi:
            if len(args.modality) == 3:
                x1_left = batch_data[f'img_{args.modality[0]}_left'].to(args.device)
                x1_right = batch_data[f'img_{args.modality[0]}_right'].to(args.device)
                x2_left = batch_data[f'img_{args.modality[1]}_left'].to(args.device)
                x2_right = batch_data[f'img_{args.modality[1]}_right'].to(args.device)
                x3_left = batch_data[f'img_{args.modality[2]}_left'].to(args.device)
                x3_right = batch_data[f'img_{args.modality[2]}_right'].to(args.device)
                out_left = model(x1_left, x2_left, x3_left)
                out_right = model(x1_right, x2_right, x3_right)
            elif len(args.modality) == 2:
                x1_left = batch_data[f'img_{args.modality[0]}_left'].to(args.device)
                x1_right = batch_data[f'img_{args.modality[0]}_right'].to(args.device)
                x2_left = batch_data[f'img_{args.modality[1]}_left'].to(args.device)
                x2_right = batch_data[f'img_{args.modality[1]}_right'].to(args.device)
                out_left = model(x1_left, x2_left)
                out_right = model(x1_right, x2_right)
            else:
                x1_left = batch_data[f'img_{args.modality[0]}_left'].to(args.device)
                x1_right = batch_data[f'img_{args.modality[0]}_right'].to(args.device)
                out_left = model(x1_left)
                out_right = model(x1_right)
            target = batch_data['label'].to(args.device)
            out = (out_left + out_right) / 2
            loss = criterion(out, target)
            # target_left = batch_data['label_left'].to(args.device)
            # target_right = batch_data['label_right'].to(args.device)
            # loss_left = criterion(out_left, target_left)
            # loss_right = criterion(out_right, target_right)
            # out = torch.sum(out, dim=0) / 2
            # out = torch.stack([out_left, out_right])
            # loss = (loss_left + loss_right) / 2
            # target = torch.stack([target_left, target_right])
            # target = torch.sum(target, dim=0) / 2
        elif len(args.modality) == 3:
            x1 = batch_data[f'img_{args.modality[0]}'].to(args.device)
            x2 = batch_data[f'img_{args.modality[1]}'].to(args.device)
            x3 = batch_data[f'img_{args.modality[2]}'].to(args.device)
            target = batch_data['label'].to(args.device)
            out = model(x1, x2, x3)
            loss = criterion(out, target)
        elif len(args.modality) == 2:
            x1 = batch_data[f'img_{args.modality[0]}'].to(args.device)
            x2 = batch_data[f'img_{args.modality[1]}'].to(args.device)
            target = batch_data['label'].to(args.device)
            if 'AMGB' in args.model:
                out, at_gen_loss = model(x1, x2)
                loss = criterion(out, target)
                # print('LOSS : ',loss, at_gen_loss)
                for at_loss in at_gen_loss:
                    loss += at_loss / 4
            else:
                out = model(x1, x2)
                loss = criterion(out, target)
        else:
            x = batch_data[f'img_{args.modality[0]}'].to(args.device)
            if 'vit' in args.model:
                out, _ = model(x)
            else:
                out = model(x)
            target = batch_data['label'].to(args.device)
            loss = criterion(out, target)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if scheduler:
            scheduler.step()
        new_cf_list = cf_matrix(y_true=target, y_pred=out, num_label=args.num_label)
        cf_list = add_element(cf_list, new_cf_list)
        loss_epoch += [loss.item()]
    model.train()
    return np.mean(loss_epoch), cf_list


def val(args, val_loader, model, criterion):
    loss_epoch = []
    cf_list = [0] * (args.num_label ** 2)
    model.eval()
    with torch.no_grad():
        for step, batch_data in enumerate(val_loader):
            if args.roi:
                if len(args.modality) == 3:
                    x1_left = batch_data[f'img_{args.modality[0]}_left'].to(args.device)
                    x1_right = batch_data[f'img_{args.modality[0]}_right'].to(args.device)
                    x2_left = batch_data[f'img_{args.modality[1]}_left'].to(args.device)
                    x2_right = batch_data[f'img_{args.modality[1]}_right'].to(args.device)
                    x3_left = batch_data[f'img_{args.modality[2]}_left'].to(args.device)
                    x3_right = batch_data[f'img_{args.modality[2]}_right'].to(args.device)
                    out_left = model(x1_left, x2_left, x3_left)
                    out_right = model(x1_right, x2_right, x3_right)
                elif len(args.modality) == 2:
                    x1_left = batch_data[f'img_{args.modality[0]}_left'].to(args.device)
                    x1_right = batch_data[f'img_{args.modality[0]}_right'].to(args.device)
                    x2_left = batch_data[f'img_{args.modality[1]}_left'].to(args.device)
                    x2_right = batch_data[f'img_{args.modality[1]}_right'].to(args.device)
                    out_left = model(x1_left, x2_left)
                    out_right = model(x1_right, x2_right)
                else:
                    x1_left = batch_data[f'img_{args.modality[0]}_left'].to(args.device)
                    x1_right = batch_data[f'img_{args.modality[0]}_right'].to(args.device)
                    out_left = model(x1_left)
                    out_right = model(x1_right)
                target = batch_data['label'].to(args.device)
                out = (out_left + out_right) / 2
                loss = criterion(out, target)
            elif len(args.modality) == 3:
                x1 = batch_data[f'img_{args.modality[0]}'].to(args.device)
                x2 = batch_data[f'img_{args.modality[1]}'].to(args.device)
                x3 = batch_data[f'img_{args.modality[2]}'].to(args.device)
                target = batch_data['label'].to(args.device)
                out = model(x1, x2, x3)
                loss = criterion(out, target)
            elif len(args.modality) == 2:
                x1 = batch_data[f'img_{args.modality[0]}'].to(args.device)
                x2 = batch_data[f'img_{args.modality[1]}'].to(args.device)
                target = batch_data['label'].to(args.device)
                if 'AMGB' in args.model:
                    out, at_gen_loss = model(x1, x2)
                    loss = criterion(out, target)
                    for at_loss in at_gen_loss:
                        loss += at_loss
                else:
                    out = model(x1, x2)
                    loss = criterion(out, target)
            else:
                x = batch_data[f'img_{args.modality[0]}'].to(args.device)
                target = batch_data['label'].to(args.device)
                if 'vit' in args.model:
                    out, _ = model(x)
                else:
                    out = model(x)
                loss = criterion(out, target)
            new_cf_list = cf_matrix(y_true=target, y_pred=out, num_label=args.num_label)
            cf_list = add_element(cf_list, new_cf_list)
            loss_epoch += [loss.item()]
        return np.mean(loss_epoch), cf_list


def pl_worker_init_function(worker_id: int) -> None:  # pragma: no cover
    """
    The worker_init_fn that Lightning automatically adds to your dataloader if you previously set
    set the seed with ``seed_everything(seed, workers=True)``.
    See also the PyTorch documentation on
    `randomness in DataLoaders <https://pytorch.org/docs/stable/notes/randomness.html#dataloader>`_.
    """

    def _get_rank() -> int:
        """Returns 0 unless the environment specifies a rank."""
        rank_keys = ("RANK", "SLURM_PROCID", "LOCAL_RANK")
        for key in rank_keys:
            rank = os.environ.get(key)
            if rank is not None:
                return int(rank)
        return 0

    global_rank = _get_rank()
    # implementation notes: https://github.com/pytorch/pytorch/issues/5059#issuecomment-817392562

    process_seed = torch.initial_seed()
    # back out the base seed so we can use all the bits
    base_seed = process_seed - worker_id
    ss = np.random.SeedSequence([base_seed, worker_id, global_rank])
    # use 128 bits (4 x 32-bit words)
    np.random.seed(ss.generate_state(4))
    # Spawn distinct SeedSequences for the PyTorch PRNG and the stdlib random module
    torch_ss, stdlib_ss = ss.spawn(2)
    # PyTorch 1.7 and above takes a 64-bit seed
    dtype = np.uint64 if torch.__version__ > "1.7.0" else np.uint32
    torch.manual_seed(torch_ss.generate_state(1, dtype=dtype)[0])
    # use 128 bits expressed as an integer
    stdlib_seed = (
            stdlib_ss.generate_state(2, dtype=np.uint64).astype(object) * [1 << 64, 1]
    ).sum()
    random.seed(stdlib_seed)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


class CNN(nn.Module):
    def __init__(self, convolutions, fc, n_labels, device, ckpt):
        super(CNN, self).__init__()
        self.convolutions = convolutions.to(device)
        self.fc = fc.to(device)
        self.n_labels = n_labels
        # self.transfer_weight(ckpt['model'])
        self.transfer_weight(ckpt)

    def layers(self):
        return nn.Sequential(self.convolutions, self.fc)

    def transfer_weight(self, ckpt):
        from collections import OrderedDict
        convolutions_dict = OrderedDict(
            [
                (k.replace("encoder.", ""), v)
                for k, v in ckpt.items()
                if "encoder" in k
            ]
        )
        self.convolutions.load_state_dict(convolutions_dict)

    def forward(self, x):
        x = self.convolutions(x)
        return self.fc(x)


def main(gpu, args):
    result_df = make_df(True)
    seed_everything(args.seed)
    if 'caps' in args.dir_data:
        min_max_transform = transforms.Compose([MinMaxNormalization()])
    else:
        min_max_transform = None

    for fold in range(5):
        print(f'Fold is {fold}')
        train_dataset = Dataset(dataset='train', fold=fold, args=args, transformation=min_max_transform)
        val_dataset = Dataset(dataset='validation', fold=fold, args=args, transformation=min_max_transform)

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=args.workers,
            pin_memory=True,
            worker_init_fn=pl_worker_init_function,
        )

        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=args.workers,
            pin_memory=True
        )

        if args.resize64:
            after_ = 500
        else:
            after_ = 1300
        if args.resize:
            flatten_shape = 128 * 2 * 2 * 3 * 2
        else:
            flatten_shape = 128 * 2 * 5 * 6 * 5

        model_2 = None
        # initializer model
        if args.model == 'Conv5FC3':
            model = Conv5_FC3(flatten_shape=128 * 3 * 3 * 3, after_=after_)
        elif args.model == 'Conv4FC3':
            # for patches
            model = Conv4_FC3(flatten_shape=flatten_shape)
        elif args.model == 'resnet':
            model = generate_model(args.model_depth)
        elif 'vit' in args.model:
            # t1_linear patch size = 13,16,15
            # ants, fsl patch size = 14 X 14
            # freesurfer 256 patch size = 8, 8, 8
            # freesurfer 128 patch size = 8, 8, 8
            import models_vit
            # model = models_vit.__dict__[args.model]()
            from monai.networks.nets import ViT
            model = ViT(in_channels=1, img_size=(128, 128, 128), patch_size=(16, 16, 16), classification=True)
        elif 'shared_multi' == args.model:
            from residual_att import shared_multimodality, shared_multimodality_roi, shared_multimodality_roi_three, \
                shared_multimodality_three
            if args.roi:
                if args.region == 'hippo':
                    if len(args.modality) == 3:
                        flatten_size = 5000 * 1 * 1 * 1
                    else:
                        # flatten_size = 512 * 3 * 3 * 3
                        flatten_size = 2048 * 1 * 1 * 1
                else:
                    if len(args.modality) == 3:
                        # flatten_size = 1000 * 5 * 6 * 5
                        flatten_size = 5000 * 2 * 3 * 2
                    else:
                        # flatten_size = 512 * 5 * 6 * 5
                        flatten_size = 2048 * 2 * 3 * 2
                if len(args.modality) == 3:
                    model = shared_multimodality_roi_three(flatten_size)
                else:
                    model = shared_multimodality_roi(flatten_size)
                # model_2 = shared_multimodality_roi(flatten_size)
            else:
                if len(args.modality) == 3:
                    model = shared_multimodality_three()
                else:
                    model = shared_multimodality()
        elif 'wide_res_unimodal' == args.model:
            if args.roi:
                from residual_att import unimodal_roi
                if args.region == 'hippo':
                    flatten_size = 216 * 3 * 3 * 3
                else:
                    flatten_size = 216 * 5 * 6 * 5
                model = unimodal_roi(flatten_size)
            else:
                from residual_att import residual_first_wide_unimodal
                model = residual_first_wide_unimodal(324 * 5 * 6 * 5 * 2)
        else:
            model = None

        if (args.device.type == 'cuda') and (torch.cuda.device_count() > 1):
            print("Multi GPU ACTIVATES")
            model = nn.DataParallel(model)

        model = model.to(args.device)
        if len(args.modality) == 3 and not args.roi:
            summary(model, ((4, 1, 160, 192, 160), (4, 1, 160, 192, 160), (4, 1, 160, 192, 160)))
        elif len(args.modality) == 3 and args.roi:
            if args.region == 'hippo':
                summary(model, ((4, 1, 50, 50, 50), (4, 1, 50, 50, 50), (4, 1, 50, 50, 50)))
            else:
                summary(model, ((4, 1, 80, 96, 80), (4, 1, 80, 96, 80), (4, 1, 80, 96, 80)))
        elif len(args.modality) == 2 and not args.roi:
            summary(model, ((4, 1, 160, 192, 160), (4, 1, 160, 192, 160)))
        elif len(args.modality) == 2 and args.roi:
            if args.region == 'hippo':
                summary(model, ((4, 1, 50, 50, 50), (4, 1, 50, 50, 50)))
            else:
                summary(model, ((4, 1, 80, 96, 80), (4, 1, 80, 96, 80)))
        else:
            if args.roi:
                if args.region == 'hippo':
                    summary(model, (4, 1, 50, 50, 50))
                else:
                    summary(model, (4, 1, 80, 96, 80))
            else:
                summary(model, (4, 1, 160, 192, 160))

        # optimizer = torch.optim.Adam(params=filter(lambda x: x.requires_grad, model.parameters()), lr=args.lr,
        #                              weight_decay=args.weight_decay)
        optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr,
                                     weight_decay=args.weight_decay)

        criterion = nn.CrossEntropyLoss()
        # criterion = SupConLoss().to(args.device)

        save_path = f'{args.model_path}/finetune_model_image_{fold}.pt'

        early_stopping = EarlyStopping(patience=args.patience, verbose=True, path=save_path, multigpu=False,
                                       pretrain=True)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs * len(train_loader))
        # scheduler = None
        for epoch in range(args.start_epoch, args.epochs):
            print(f"Epoch : {epoch}")
            model.train()

            loss_epoch, train_cf_list = train(args, train_loader, model, criterion, optimizer, scheduler)
            train_acc, train_bacc, train_spe, train_sen, train_ppv, train_npv, train_f1 = cal_metric(train_cf_list)
            val_loss_epoch, val_cf_list = val(args, val_loader, model, criterion)
            val_acc, val_bacc, val_spe, val_sen, val_ppv, val_npv, val_f1 = cal_metric(val_cf_list)
            # accuracy, balnced accuracy, specificity, sensitivity, F1, precision
            print('\n')
            print('\n')
            print(f"Fold {fold}: epoch : {epoch}")
            print(f'Loss : Train = {loss_epoch} | Val = {val_loss_epoch} ')
            print(
                f"Train  ||  Acc : {train_acc:.4f} BACC: {train_bacc:.4f} SPE : {train_spe:.4f} SEN : {train_sen:.4f} PPV : {train_ppv:.4f} F1 : {train_f1:.4f}")
            print(
                f"Val  ||  Acc : {val_acc:.4f} BACC: {val_bacc:.4f} SPE : {val_spe:.4f} SEN : {val_sen:.4f} PPV : {val_ppv:.4f} F1 : {val_f1:.4f}")
            # print(f'ACC : {np.round(train_acc, 4)} ')
            # print(f'BACC : {np.round(train_bacc, 4)}')
            # print(f'Specificity : {np.round(train_spe, 4)} | Sensitivity : {np.round(train_sen, 4)}')
            # print(f'PPV : {np.round(train_ppv, 4)}')
            # print(f'NPV : {np.round(train_npv, 4)}')
            # print(f'F1 : {np.round(train_f1, 4)}')
            # print('\n')
            # print("Validation")
            # print('\n')
            # print(f'ACC : {np.round(val_acc, 4)}')
            # print(f'BACC : {np.round(val_bacc, 4)}')
            # print(f'Specificity : {np.round(val_spe, 4)} | Sensitivity : {np.round(val_sen, 4)}')
            # print(f'PPV : {np.round(val_ppv, 4)}')
            # print(f'NPV : {np.round(val_npv, 4)}')
            # print(f'F1 : {np.round(val_f1, 4)}')
            early_stopping(val_loss=None, model=model, f1=val_bacc)
            output_list = [epoch, loss_epoch, val_loss_epoch, train_acc, val_acc, train_bacc, val_bacc, train_spe,
                           val_spe, train_sen, val_sen, train_ppv, val_ppv, train_npv, val_npv, train_f1, val_f1]

            result_df = write_result(output_list, result_df)

            if epoch == args.epochs or early_stopping.early_stop:
                break
        if len(args.modality) == 3:
            result_df.to_csv(
                f'{args.model_path}/finetune_{args.modality[0]}_{args.modality[1]}_{args.modality[2]}_cnmci_results_fold_{fold}.csv')
        elif len(args.modality) == 2:
            result_df.to_csv(
                f'{args.model_path}/finetune_{args.modality[0]}_{args.modality[1]}_cnmci_results_fold_{fold}.csv')
        else:
            result_df.to_csv(f'{args.model_path}/finetune_{args.modality[0]}_cnmci_results_fold_{fold}.csv')


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="SimCLR")
    parser.add_argument('--config', default='./config/config.yaml', type=str)

    args = parser.parse_args()
    config = yaml_config_hook(args.config)

    for k, v in config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))
    args = parser.parse_args()

    for arg in vars(args):
        print(f'--{arg}', getattr(args, arg))

    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)

    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    main(0, args)
