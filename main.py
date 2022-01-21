import argparse
import os
import random

import numpy as np
import tensorboardX
import torch
import torchvision
from prefetch_generator import BackgroundGenerator
# from pudb import set_trace
from torch import distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
from tqdm import tqdm

from metrics import AverageMeter, ExpStat


class DataLoaderX(DataLoader):
    """(加速组件) 重新封装Dataloader，使prefetch不用等待整个iteration完成"""

    def __iter__(self):
        return BackgroundGenerator(super().__iter__(), max_prefetch=10)


def main(args):
    #######################################################################
    # Initialize DDP setting
    #######################################################################

    if args.local_rank != -1:
        torch.cuda.set_device(args.local_rank)
        dist.init_process_group(backend='nccl')
        args.world_size = dist.get_world_size()

    if args.local_rank in [-1, 0]:
        # 初始化实验记录工具
        writer = tensorboardX.SummaryWriter(log_dir="./log/")

    #######################################################################
    # Initialize Dataset and Dataloader
    #######################################################################
    transform = {
        "train":
        transforms.Compose([
            transforms.RandomCrop(size=(32, 32), padding=4),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ]),
        "val":
        transforms.Compose([
            transforms.Resize(size=(32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])
    }

    trainset = torchvision.datasets.CIFAR10(root="./data/cifar10",
                                            train=True,
                                            transform=transform["train"],
                                            download=True)
    train_num_samples_per_cls = [
        trainset.targets.count(i) for i in range(len(trainset.classes))
    ]
    valset = torchvision.datasets.CIFAR10(root="./data/cifar10",
                                          train=False,
                                          transform=transform["val"],
                                          download=True)
    val_num_samples_per_cls = [
        valset.targets.count(i) for i in range(len(valset.classes))
    ]
    # DistributedSampler 负责数据分发到多卡

    if args.local_rank != -1:
        args.batch_size = int(args.batch_size / args.world_size)
        train_sampler = DistributedSampler(trainset)
        val_sampler = DistributedSampler(valset)
    else:
        train_sampler = None
        val_sampler = None

    train_loader = DataLoaderX(
        trainset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=(train_sampler is None),
        pin_memory=True,
        drop_last=True,
        sampler=train_sampler,
    )

    val_loader = DataLoaderX(
        valset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=(val_sampler is None),
        pin_memory=True,
        drop_last=False,
        sampler=val_sampler,
    )

    #######################################################################
    # 初始化网络模型
    #######################################################################

    if args.local_rank in [-1, 0]:
        print("Initializing Model...")
    model = torchvision.models.resnet18(
        num_classes=len(trainset.classes)).cuda()

    #######################################################################
    # 初始化 Loss
    #######################################################################

    if args.local_rank in [-1, 0]:
        print("Initializing Criterion...")
    criterion = torch.nn.CrossEntropyLoss(weight=None).cuda()

    #######################################################################
    # 初始化 Optimizer
    #######################################################################

    if args.local_rank in [-1, 0]:
        print("Initializing Optimizer...")
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=args.lr,
                                momentum=0.9,
                                nesterov=True)

    #######################################################################
    # 初始化 DistributedDataParallel
    #######################################################################

    if args.local_rank != -1:
        if args.local_rank in [-1, 0]:
            print("Initializing DistributedDataParallel...")
        model = DistributedDataParallel(model,
                                        device_ids=[args.local_rank],
                                        output_device=args.local_rank)

    #######################################################################
    # 初始化 LR Scheduler
    #######################################################################

    if args.local_rank in [-1, 0]:
        print("Initializing lr_scheduler...")
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=30,
                                                   gamma=0.1)

    #######################################################################
    # 开始训练
    #######################################################################

    if args.local_rank in [-1, 0]:
        print(f"\nStart {args.total_epoch}-epoch training ...\n")

    for epoch in range(args.total_epoch):
        if args.local_rank != -1:
            # DistributedSampler 需要在每个epoch 打乱顺序分配到各卡, 为了同步
            # 各个卡组成的数据互补，则采用epoch为seed，生成相同的序列，让各个
            # 程序各取所需，详情看源代码
            train_sampler.set_epoch(epoch)
            val_sampler.set_epoch(epoch)
            dist.barrier()

        train_stat, train_loss = train_epoch(
            epoch=epoch,
            train_loader=train_loader,
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            num_samples_per_cls=train_num_samples_per_cls,
            args=args,
        )
        val_stat, val_loss = eval_epoch(
            epoch=epoch,
            val_loader=val_loader,
            model=model,
            criterion=criterion,
            num_samples_per_cls=val_num_samples_per_cls,
            args=args,
        )

        # Record the experimental result.
        # - tensorboard.SummaryWriter
        # - logging

        if args.local_rank in [-1, 0]:
            writer.add_scalars("Loss", {
                "train": train_loss,
                "val": val_loss
            }, epoch)
            writer.add_scalars("Acc", {
                "train": train_stat.acc,
                "val": val_stat.acc
            }, epoch)

        lr_scheduler.step()

    print("End Experiments.")


def train_epoch(epoch, train_loader, model, criterion, optimizer, lr_scheduler,
                num_samples_per_cls, args):
    model.train()

    train_loss_meter = AverageMeter()
    train_stat = ExpStat(num_samples_per_cls)

    if args.local_rank in [-1, 0]:
        train_pbar = tqdm(total=len(train_loader),
                          desc=f"Train Epoch[{epoch:>3d}/{args.total_epoch}]")

    for i, (batch_imgs, batch_targets) in enumerate(train_loader):
        optimizer.zero_grad()

        batch_imgs, batch_targets = batch_imgs.cuda(), batch_targets.cuda()

        batch_probs = model(batch_imgs)
        batch_avg_loss = criterion(batch_probs, batch_targets)

        if args.local_rank != -1:
            dist.barrier()
            # torch.distributed.barrier()的作用是，阻塞进程
            # 确保每个进程都运行到这一行代码，才能继续执行，这样计算
            # 平均loss和平均acc的时候，不会出现因为进程执行速度不一致
            # 而导致错误
            batch_avg_loss.backward()
            optimizer.step()
            batch_avg_loss = _reduce_tensor(batch_avg_loss, args.world_size)
        else:
            batch_avg_loss.backward()
            optimizer.step()

        train_loss_meter.update(batch_avg_loss.item())
        batch_preds = torch.argmax(batch_probs, dim=1)
        train_stat.update(batch_targets, batch_preds)

        if args.local_rank in [-1, 0]:
            train_pbar.update()
            train_pbar.set_postfix_str(
                f"LR:{optimizer.param_groups[0]['lr']:.1e} "
                f"Loss:{train_loss_meter.avg:>3.1f}")

    if args.local_rank != -1:
        # all reduce the statistical confusion matrix
        dist.barrier()
        # 统计所有进程的train_stat里的confusion matrix
        # 由于ddp通信只能通过tensor, 所以这里采用cm，信息最全面，
        # 可操作性强
        train_stat._cm = _reduce_tensor(train_stat._cm,
                                        args.world_size,
                                        op='sum')

    lr_scheduler.step()  # 如果是iter更新，则放入内循环

    if args.local_rank in [-1, 0]:
        train_pbar.set_postfix_str(f"LR:{optimizer.param_groups[0]['lr']:.1e} "
                                   f"Loss:{train_loss_meter.avg:>4.2f} "
                                   f"MR:{train_stat.mr:>6.2%} ")

    return train_stat, train_loss_meter.avg


def eval_epoch(epoch, val_loader, model, criterion, num_samples_per_cls, args):
    model.eval()

    val_loss_meter = AverageMeter()
    val_stat = ExpStat(num_samples_per_cls)

    if args.local_rank in [-1, 0]:
        val_pbar = tqdm(total=len(val_loader),
                        ncols=0,
                        desc="             Eval")

    for i, (batch_imgs, batch_targets) in enumerate(val_loader):

        batch_imgs, batch_targets = batch_imgs.cuda(), batch_targets.cuda()

        batch_probs = model(batch_imgs)
        batch_avg_loss = criterion(batch_probs, batch_targets)

        if args.local_rank != -1:
            dist.barrier()
            # torch.distributed.barrier()的作用是，阻塞进程
            # 确保每个进程都运行到这一行代码，才能继续执行，这样计算
            # 平均loss和平均acc的时候，不会出现因为进程执行速度不一致
            # 而导致错误
            batch_avg_loss = _reduce_tensor(batch_avg_loss, args.world_size)

        val_loss_meter.update(batch_avg_loss.item())
        batch_preds = torch.argmax(batch_probs, dim=1)
        val_stat.update(batch_targets, batch_preds)

        if args.local_rank in [-1, 0]:
            val_pbar.update()
            val_pbar.set_postfix_str(f"Loss:{val_loss_meter.avg:>3.1f}")

    if args.local_rank != -1:
        # all reduce the statistical confusion matrix
        dist.barrier()
        # 统计所有进程的train_stat里的confusion matrix
        # 由于ddp通信只能通过tensor, 所以这里采用cm，信息最全面，
        # 可操作性强
        val_stat._cm = _reduce_tensor(val_stat._cm, args.world_size, op='sum')

    if args.local_rank in [-1, 0]:
        val_pbar.set_postfix_str(f"Loss:{val_loss_meter.avg:>4.2f} "
                                 f"MR:{val_stat.mr:>6.2%} ")

    return val_stat, val_loss_meter.avg


def _reduce_tensor(tensor, nproc, op='mean'):
    reduced_tensor = tensor.clone()
    dist.all_reduce(reduced_tensor, op=dist.ReduceOp.SUM)

    if op == 'mean':
        reduced_tensor /= nproc

    return reduced_tensor


def _set_random_seed(seed=0, cuda_deterministic=False):
    """Set seed and control the balance between reproducity and efficiency

    Reproducity: cuda_deterministic = True
    Efficiency: cuda_deterministic = False
    """

    random.seed(seed)
    np.random.seed(seed)

    assert torch.cuda.is_available()
    torch.manual_seed(seed)  # sets the seed for generating random numbers.
    torch.cuda.manual_seed_all(seed)

    if cuda_deterministic:  # slower, but more reproducible
        torch.backends.cudnn.enabled = False
        torch.backends.cudnn.deterministic = True  # 固定内部随机性
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True  # 输入尺寸一致，加速训练


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_rank',
                        type=int,
                        default=-1,
                        help='Local Rank for distributed training. '
                        'if single-GPU, default: -1')
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--total-epoch", type=int, default=100)

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()

    if "LOCAL_RANK" in os.environ:
        args.local_rank = int(os.environ["LOCAL_RANK"])

    _set_random_seed(args.seed)
    main(args)
