import os
import yaml
import torch

def workspace_config(opt):
    workspace = os.getenv('workspace')
    if workspace is not None:
        os.makedirs(workspace, exist_ok=True)
        opt.out_dir = os.path.join(workspace, opt.out_dir)
        opt.log_dir = os.path.join(workspace, opt.log_dir)
    return opt

def train_config(opt):
    yaml_path = opt.config
    with open(yaml_path, 'r') as f:
        info = yaml.load(f, Loader=yaml.FullLoader)
    for key in info.keys():
        value = info[key]
        setattr(opt, key, value)
    return opt

def init_distributed_mode(args):
    """ init for distribute mode """
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.local_rank = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.local_rank = args.rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.local_rank)
    '''
    This is commented due to the stupid icoding pylint checking.
    print('distributed init rank {}: {}'.format(args.rank, args.dist_url), flush=True)
    '''
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                        world_size=args.world_size, rank=args.rank)
    torch.distributed.barrier()

    if args.distributed:
        print('[INFO] turn on distributed train', flush=True)
    else:
        print('[INFO] turn off distributed train', flush=True)