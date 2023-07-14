from dataset import create_dataset
from option import get_opt
from model import lumos
from tqdm import tqdm
from utils import init_distributed_mode
from torch.utils.tensorboard import SummaryWriter
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

if __name__ == "__main__":
    opt = get_opt()
    init_distributed_mode(opt)
    dataloader_train, dataloader_val = create_dataset(opt)
    model = lumos(opt)

    # configure ddp
    get_model = lambda model: model.module if opt.distributed else model
    if opt.distributed:
        model = DDP(model.cuda(), device_ids=[opt.local_rank], output_device=opt.local_rank)
        if dist.get_rank() == 0:
            writer = SummaryWriter(opt.out_dir)
    else:
        writer = SummaryWriter(opt.out_dir)
    start_epoch = get_model(model).load_ckpt()

    for epoch in tqdm(range(start_epoch, opt.epochs + 1), ascii=True, desc='epoch'):
        if opt.distributed:
            dataloader_train.sampler.set_epoch(epoch)
        for i, data in tqdm(enumerate(dataloader_train), ascii=True, desc='training iterations'):
            get_model(model).set_input(data)
            get_model(model).optimize_parameters()
            if (opt.distributed and dist.get_rank() == 0) or not opt.distributed:
                get_model(model).gather_loss()

        get_model(model).update_lr(epoch)
        if (opt.distributed and dist.get_rank() == 0) or not opt.distributed:
            # print losses and add them to writer
            loss = get_model(model).gather_loss(True)
            out = "[Epoch {} ]".format(epoch)
            for idx, key in enumerate(loss.keys()):
                average = loss[key] / (i + 1)
                writer.add_scalar(key, average, epoch)
                out += (key + ": " + str(average) + ("\n" if idx == len(loss.keys()) - 1 else " "))
            tqdm.write(out)
            
            # save ckpt
            if epoch % opt.save_fre == 0:
                get_model(model).save_ckpt(epoch)

            if epoch % opt.val_fre == 0:
                for i, data in enumerate(dataloader_val):
                    with torch.no_grad():
                        get_model(model).set_input(data, val=True)
                        get_model(model).eval()
                        get_model(model).forward(val=True)
                        get_model(model).plot_val(epoch)
                        get_model(model).train()