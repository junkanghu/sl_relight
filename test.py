from dataset import create_dataset, Dataset, get_dir
from option import get_opt
from model import lumos
from tqdm import tqdm, trange
from tensorboardX import SummaryWriter
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

import cv2
import numpy as np


if __name__ == "__main__":
    opt = get_opt()
    opt.test = True
    model = lumos(opt)
    model.load_ckpt(True)
    dataloader_test = create_dataset(opt)
    for i, data in enumerate(dataloader_test):
        with torch.no_grad():
            model.set_input(data, test=True)
            model.eval()
            model.forward()
            model.plot_val(i, test='test')