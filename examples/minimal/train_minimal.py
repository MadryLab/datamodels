from fastargs import get_current_config
from fastargs.decorators import param
from fastargs import Param, Section
from fastargs.validation import And, OneOf
from argparse import ArgumentParser
import numpy as np

Section('trainer').params(
    multiple=Param(int, 'multiple', required=True),
)

def main(*_, index, logdir):
    make_config()
    return execute(index=index)

@param('trainer.multiple')
def execute(*_, index, multiple):
    return {
        'random_numbers': np.ones(256, dtype=np.uint8) * multiple
    }

def make_config(quiet=False):
    config = get_current_config()
    parser = ArgumentParser()
    config.augment_argparse(parser)
    config.collect_argparse_args(parser)
    config.validate(mode='stderr')
    if not quiet:
        config.summary()