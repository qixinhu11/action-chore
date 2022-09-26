from ast import arg
import sys, os
sys.path.append(os.getcwd())
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from utils.dist_utils import init_distributed_mode

from model import ACTIONCHORE
from trainer.trainer import Trainer
from data.data_paths import DataPaths
from data.train_data import BehaveDataset



def test(args):
    world_size = torch.cuda.device_count()
    init_distributed_mode(args)
    rank = args.rank
    device = torch.device(args.device)
    model = ACTIONCHORE(args, rank=rank)

    # test images 
    # net_img_size = 512, 512
    # batch_size = 15
    test_img = torch.randn((15, 5, 512, 512))  # [B, C, H, W] 
    model.filter(test_img)


def prepare_img():
    # img_size 512, 512
    # crop_size 1200
    pass 

if __name__ == "__main__":
    from argparse import ArgumentParser
    from config.config_loader import load_configs
    parser = ArgumentParser()
    parser.add_argument('-en', '--exp_name')

    # multi-gpu arguments
    # device will be set by system sutomatically
    parser.add_argument('--device', default='cuda', help='device id (i.e. 0 or 0,1 or cpu)')
    # number of processes, i.e. number of GPUs
    parser.add_argument('-w', '--world_size', default=4, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

    # for pycharm debug
    parser.add_argument('-d1', )
    parser.add_argument('--multiproc')
    parser.add_argument('--qt-support')

    args = parser.parse_args()

    configs = load_configs(args.exp_name)
    assert args.exp_name==configs.exp_name

    # add command line configs
    configs.device = args.device
    configs.world_size = args.world_size
    configs.dist_url = args.dist_url

    test(configs)
