import hashlib
import os
import sys
from typing import Any, Dict

import gin
import pytorch_lightning as pl
import torch
from absl import flags
from torch.utils.data import DataLoader

import rave
import rave.core
import rave.dataset

# fix torch device order to be same as nvidia-smi order
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

FLAGS = flags.FLAGS

flags.DEFINE_string('name', None, help='Name of the run', required=True)
flags.DEFINE_multi_string('config',
                          default='v2.gin',
                          help='RAVE configuration to use')
flags.DEFINE_string('db_path',
                    None,
                    help='Preprocessed dataset path',
                    required=True)
flags.DEFINE_integer('max_steps',
                     6000000,
                     help='Maximum number of training steps')
flags.DEFINE_integer('val_every', 10000, help='Checkpoint model every n steps')
flags.DEFINE_integer('n_signal',
                     126976,
                     help='Number of audio samples to use during training')
flags.DEFINE_integer('batch', 8, help='Batch size')
flags.DEFINE_string('ckpt',
                    None,
                    help='Path to checkpoint to continue training from')
flags.DEFINE_string('transfer_ckpt',
                    None,
                    help='Path to checkpoint to initialize weights from')
flags.DEFINE_multi_string('override', default=[], help='Override gin binding')
flags.DEFINE_integer('workers',
                     default=8,
                     help='Number of workers to spawn for dataset loading')
flags.DEFINE_multi_integer('gpu', default=None, help='GPU to use')
flags.DEFINE_bool('derivative',
                  default=False,
                  help='Train RAVE on the derivative of the signal')
flags.DEFINE_bool('normalize',
                  default=False,
                  help='Train RAVE on normalized signals')
flags.DEFINE_float('speed_semitones',
                  default=0,
                  help='speed change data augmentation')
flags.DEFINE_float('gain_db',
                  default=0,
                  help='gain change data augmentation')
flags.DEFINE_float('allpass_p',
                  default=0.8,
                  help='chance of allpass filter data augmentation')
flags.DEFINE_float('eq_p',
                  default=0,
                  help='chance of random EQ data augmentation')
flags.DEFINE_float('delay_p',
                  default=0,
                  help='chance of random comb delay data augmentation')
flags.DEFINE_float('distort_p',
                  default=0,
                  help='chance of random distortion data augmentation')
flags.DEFINE_float('ema',
                   default=None,
                   help='Exponential weight averaging factor (optional)')
flags.DEFINE_bool('progress',
                  default=True,
                  help='Display training progress bar')
