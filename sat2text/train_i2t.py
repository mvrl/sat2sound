import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
import numpy as np
import torch
import os
import random
from argparse import ArgumentParser
import sys
import warnings
import yaml
from .engine_i2t import sat2textModel
from src.config import cfg
torch.autograd.set_detect_anomaly(os.environ.get("DETECT_ANOMALY", "0") == "1")
# os.environ["TORCH_CPP_LOG_LEVEL"]="INFO"
# os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"

if not sys.warnoptions:
    warnings.filterwarnings("ignore", category=UserWarning, module="pytorch_lightning")
    warnings.filterwarnings("ignore", category=UserWarning, module="torch")

os.environ["WANDB__SERVICE_WAIT"] = "300"


def set_seed(seed: int = 56) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")


def _apply_yaml_config(parser, argv):
    """If --config <path> is present in argv, load YAML and use as argparse defaults.
    CLI flags still override YAML values. Unknown keys raise to catch typos early."""
    config_parser = ArgumentParser(add_help=False)
    config_parser.add_argument('--config', type=str, default=None)
    pre_args, _ = config_parser.parse_known_args(argv)
    if pre_args.config is None:
        return
    with open(pre_args.config) as f:
        cfg_dict = yaml.safe_load(f) or {}
    valid_dests = {a.dest for a in parser._actions}
    unknown = set(cfg_dict) - valid_dests
    if unknown:
        raise ValueError(f"Unknown keys in {pre_args.config}: {sorted(unknown)}")
    parser.set_defaults(**cfg_dict)


def get_args():
    parser = ArgumentParser(description='')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to YAML config; values become defaults, CLI flags override.')
    #training hparams
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--limit_val_batches', type=int, default=100)
    parser.add_argument('--val_check_interval', type=int, default=500)
    parser.add_argument('--batch_size', type=int, default=128) 
    
    parser.add_argument('--max_epochs', type=int, default=25)
    parser.add_argument('--mode', type=str, default='dev', choices=['dev', 'train'])   
    
    parser.add_argument('--dataset_type', type=str, default='GeoSound_bingmap',choices=['GeoSound_sentinel','GeoSound_bingmap', 'SoundingEarth'])
    parser.add_argument('--sat_input_size', type=int, default= 224)

    parser.add_argument('--sat_type', type=str, default='bingmap', choices=['sentinel','bingmap','googleEarth'])

    parser.add_argument('--fc_dim', type=int, default = 1024)
    parser.add_argument('--codebook_dim', type=int, default = 1024)
    parser.add_argument('--codebook_size', type=int, default = 16000)

    parser.add_argument('--learning_rate', type=float, default=5e-5)
    parser.add_argument('--weight_decay', type=float, default=0.2)
    parser.add_argument('--warm_up_iterations', type=int, default=5000)
    parser.add_argument('--strategy', type=str, default='auto')

    parser.add_argument('--accelerator',type=str, default='gpu')
    parser.add_argument('--devices', type=int, default=1)
    
    parser.add_argument('--project_name', type=str, default='sat2text_fdt')
    parser.add_argument('--run_name', type=str, default='debug')
    parser.add_argument('--wandb_mode', type=str, default='disabled')
    
    parser.add_argument('--pseudo_match_alpha', type=float, default=0.1)
    
    parser.add_argument('--recall_at', type=int, default = 10) #percent
    # Training resuming parameters:
    parser.add_argument('--ckpt_path',type=str, default ='none')
    parser.add_argument('--ckpt_mode',type=str, default ='hard')

    _apply_yaml_config(parser, sys.argv[1:])
    args = parser.parse_args()

    return args




if __name__ == '__main__':
    set_seed(56)
    args = get_args()
    #set learning rate logger
    print('Starting Training')
    print(args)
    if args.mode == "dev":
        args.batch_size = 2
        args.wandb_mode = "disabled"
        # args.accelerator = "cpu"
    
    
    if args.dataset_type == "SoundingEarth":
        args.sat_type  = "googleEarth"
    else:
        args.sat_type = args.dataset_type.split("_")[1]
    
    args.satmae_ckpt_path = cfg.satmae_ckpt_path
    #initialize model
    sat2text_model = sat2textModel(args)
    #initialize checkpoints and loggers
    lr_logger = LearningRateMonitor(logging_interval='step')
    wb_logger = WandbLogger(save_dir=cfg.log_dir,project=args.project_name, name=args.run_name, mode=args.wandb_mode)
    ckpt_monitor1 = ((
            ModelCheckpoint(monitor='val_loss', mode='min', filename='{epoch}-{step}-{val_loss:.3f}',save_top_k = 3, save_last=True,save_on_train_epoch_end=False)
        ))
    ckpt_monitor2 = ((
            ModelCheckpoint(monitor='I2T_Recall', mode='max',filename='{epoch}-{step}-{I2T_Recall:.3f}',save_top_k = 3, save_last=True,save_on_train_epoch_end=False)
        ))
    

    if args.mode == 'dev': 
        print('Development Test Run')
        trainer = pl.Trainer(profiler="simple",precision=32,fast_dev_run=20, max_epochs=4, logger=wb_logger, strategy=args.strategy, num_sanity_val_steps=4,
        accelerator=args.accelerator, devices=args.devices, callbacks=[ckpt_monitor1, ckpt_monitor2, lr_logger])
    elif args.mode == 'train':
        print('Training Run')
        trainer = pl.Trainer(precision=32, max_epochs=args.max_epochs, logger=wb_logger, strategy=args.strategy, num_sanity_val_steps=0, 
        accelerator=args.accelerator, devices=args.devices, callbacks=[ckpt_monitor1, ckpt_monitor2, lr_logger], 
        val_check_interval=args.val_check_interval, check_val_every_n_epoch=None, limit_val_batches=args.limit_val_batches,
        log_every_n_steps=15)
    else:
        raise ValueError('Invalid value for mode')
    
    if args.ckpt_path.lower()=='none'.lower():
        trainer.fit(sat2text_model)
    else:
        if args.ckpt_mode.lower()=='hard':
            print('Hard Checkpoint Reload')
            trainer.fit(sat2text_model, ckpt_path=args.ckpt_path)
        elif args.ckpt_mode.lower()=='soft':
            print('Soft Checkpoint Reload')
            checkpoint = torch.load(args.ckpt_path, map_location="cpu", weights_only=False)
            sat2text_model.load_state_dict(checkpoint['state_dict'])
            trainer.fit(sat2text_model)