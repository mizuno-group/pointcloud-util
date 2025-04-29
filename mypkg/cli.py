# -*- coding: utf-8 -*-
"""

CLI template

@author: mizuno-group
"""
# packages installed in the current environment
import os
import datetime
import argparse
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm, trange

# original packages in src
from .src import utils
from .src import data_handler as dh
from .src.trainer import Trainer
from .src.models import MyNet

# argumentの設定, 概ね同じセッティングの中で振りうる条件を設定
parser = argparse.ArgumentParser(description='CLI template')
parser.add_argument(
    'workdir', type=str,
    help='working directory that contains the dataset'
    )
parser.add_argument('--note', type=str, help='short note for this running')
parser.add_argument('--seed', type=str, default=222) # seed

args = parser.parse_args() # Namespace object

# argsをconfigに変換
cfg = vars(args)

# seedの固定
utils.fix_seed(seed=args.seed, fix_gpu=False) # for seed control

# setup
now = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
cfg["outdir"] = args.workdir + '/results/' + now # for output
if not os.path.exists(cfg["outdir"]):
    os.makedirs(cfg["outdir"])

# データの準備
def prepare_data():
    """ dataの準備 """
    raise NotImplementedError

# model等の準備
def prepare_model():
    """ modelの準備 """
    raise NotImplementedError

# 学習
def fit():
    """ 学習 """
    raise NotImplementedError

# main関数, シェルで呼び出せるのでここで書く
def main():
    # training mode
    start = time.time() # for time stamp
    # 1. data prep
    data = prepare_data()
    # 2. model prep
    model = prepare_model()
    # 3. training
    train_loss, test_loss, accuracies = fit(model, data, cfg)
    # 4. modify config
    elapsed_time = utils.timer(start) # for time stamp
    cfg["elapsed_time"] = elapsed_time
    # 5. save experiment & config
    print(">> Done training")


# cli.pyを実行する
if __name__ == '__main__':
    main()