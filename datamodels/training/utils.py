from fastargs import get_current_config
from fastargs.decorators import param
from fastargs import Param, Section
from fastargs.validation import And, OneOf
from argparse import ArgumentParser
from fastargs.config import Config
import types
from pathlib import Path
import numpy as np

def memmap_path(logdir, key):
    return Path(logdir) / f'{key}.npy'

def load_memmap(logdir, key, mode):
    x = np.load(memmap_path(logdir, key), mmap_mode=mode)
    return x

def collect_known_args(self, parser, disable_help=False):
    args, _ = parser.parse_known_args()
    for fname in args.config_file:
        self.collect_config_file(fname)

    args = vars(args)
    self.collect(args)
    self.collect_env_variables()

def make_config(quiet=False, conf_path=None):
    config = get_current_config()
    if conf_path is not None:
        config.collect_config_file(conf_path)

    f = types.MethodType(collect_known_args, config)
    config.collect_argparse_args = f

    parser = ArgumentParser()
    config.augment_argparse(parser)
    config.collect_argparse_args(parser)

    config.validate(mode='stderr')
    if not quiet:
        config.summary()

    return config
