import argparse
import torch
from torch.utils.data import DataLoader
from prefetch_generator import BackgroundGenerator
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler
from apex import amp


class DataLoaderX(DataLoader):
    """(加速组件) 重新封装Dataloader，使prefetch不用等待整个iteration完成"""

    def __iter__(self):
        return BackgroundGenerator(super().__iter__(), max_prefetch=10)


def main(args):
    #######################################################################
    # Initialize DDP setting
    #######################################################################
    if args.local_rank != -1:
        torch.distributed.init_process_group(backend='nccl',
                                             init_method='env://')
        torch.cuda.set_device(args.local_rank)
        args.global_rank = torch.distributed.get_rank()
        args.world_size = torch.distributed.get_world_size()

    if args.local_rank in [-1, 0]:
        # TODO: 初始化实验记录工具
        pass

    #######################################################################
    # Initialize Dataset and Dataloader
    #######################################################################
    train_dataset = build_train_dataset()  # TODO

    # DistributedSampler 负责数据分发到多卡
    train_sampler = DistributedSampler(train_dataset) \
        if args.local_rank != -1 else None

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=(train_sampler is None),
        pin_memory=True,
        drop_last=True,
        sampler=train_sampler,
    )

    # 验证过程，只用了一张卡；
    # 熟悉DDP后，可以转向torch.distributed.all_reduce(), 进行多卡验证评估
    if args.local_rank in [-1, 0]:
        # val_dataset = build_val_dataset()  # TODO
        # val_dataloader = DataLoader(
        #     val_dataset,
        #     batch_size=args.val_batch_size,
        #     pin_memory=True,
        #     drop_last=False
        # )
        pass

    #######################################################################
    # 初始化网络模型
    #######################################################################
    model = build_model()  # TODO: include resume process

    #######################################################################
    # 初始化 Loss
    #######################################################################
    criterion = build_criterion()  # TODO

    #######################################################################
    # 初始化 Optimizer
    #######################################################################
    optimizer = build_optimizer()  # TODO: include resume process

    #######################################################################
    # 初始化 DistributedDataParallel
    #######################################################################
    if args.local_rank != -1:
        model, optimizer = amp.initialize(model, optimizer, opt_level='O1')
        model = DistributedDataParallel(
            model,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            find_unused_parameters=True
        )

    #######################################################################
    # 初始化 LR Scheduler
    #######################################################################
    lr_scheduler = init_lr_scheduler()  # TODO

    #######################################################################
    # 开始训练
    #######################################################################
    for epoch in range(args.total_epochs):
        if args.local_rank != -1:
            # DistributedSampler 需要在每个epoch 打乱顺序分配到各卡
            train_sampler.set_epoch(epoch)

        for i, (batch_imgs, batch_targets) in enumerate(train_dataloader):
            optimizer.zero_grad()

            batch_imgs, batch_targets = batch_imgs.cuda(), batch_targets.cuda()
            batch_probs = model(batch_imgs)

            batch_avg_loss = criterion(batch_probs, batch_targets)
            if args.local_rank != -1:
                with amp.scale_loss(batch_avg_loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                optimizer.step()
                with torch.no_grad():
                    # 可以去查一下 ring all reduce， 用于多卡通信，此处为求和
                    torch.distributed.all_reduce(
                        batch_avg_loss, op=torch.distributed.reduce_op.SUM
                    )

                    if args.local_rank == 0:
                        batch_avg_loss /= args.world_size

            else:
                batch_avg_loss.backward()
                optimizer.step()

            batch_preds = batch_probs.max(1)[1]

            # AverageMeter实时评估实验结果

        if args.local_rank in [-1, 0]:
            # 推理验证集
            # logging或tensorboard记录实验结果，保存断点
            pass

        lr_scheduler.step()  # 如果是iter更新，则放入内循环


def parse_args():
    parser = argparse.ArgumentParser()
    # 采用env模式启动ddp, args.local_rank参数会自动读取，不能修改
    parser.add_argument('--local_rank', type=int, help='Local Rank for\
                        distributed training. if single-GPU, default: -1')
    # 添加其他的参数，如batch_size
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    main(args)
