# in this file:
# - take an index
# - take a command to run
# - take an out directory

# - run main() which returns dict mapping: [output] -> [value]
# - run for each output key, find mmap in out directory, write to mmap[index]
#   with the value

from fastargs.decorators import param
from fastargs import Param, Section
import tqdm
from functools import cache

import importlib.util
import sys
from pathlib import Path
from .utils import memmap_path, make_config
from .spec import COMPLETED

import numpy as np
import signal
import sys

def alarm_handler(signal, frame):
    print('!' * 80)
    print('Exiting due to timeout!!')
    print('!' * 80)
    sys.exit(0)

signal.signal(signal.SIGALRM, alarm_handler)

INDEX_NONCE = -99999999
LOGDIR_NONCE = ''

Section('worker').params(
    index=Param(int, 'index of this job', default=INDEX_NONCE),
    main_import=Param(str, 'relativer python import module with main() to run',
                      required=True),
    logdir=Param(str, 'file with main() to run', default=LOGDIR_NONCE),
    do_if_complete=Param(bool, 'ignore the completed array, and do the task anyways', is_flag=True),
    job_timeout=Param(int, 'seconds per job', default=99999999)
)

@param('worker.logdir')
def kv_log(k, v, logdir, index):
    this_filename = memmap_path(logdir, k)
    # make sure that the mmap actually exists
    assert this_filename.exists(), this_filename

    # now acutally open it for writing
    mmap = get_mmap(this_filename, 'r+')
    mmap[index] = v
    mmap.flush()

@cache
def get_mmap(fn, mode):
    return np.load(fn, mmap_mode=mode)

def kv_read(k, logdir, index):
    this_filename = memmap_path(logdir, k)
    # make sure that the mmap actually exists
    assert this_filename.exists(), this_filename
    mmap = get_mmap(this_filename, 'r')
    return mmap[index]

@param('worker.index')
@param('worker.logdir')
@param('worker.do_if_complete')
@param('worker.job_timeout')
def do_index(*_, index, routine, logdir, do_if_complete, job_timeout):
    logdir = Path(logdir)

    print("logging in", logdir)
    worker_logs = Path(logdir) / 'workers' / str(index)
    worker_logs.mkdir(exist_ok=True, parents=True)

    if (not do_if_complete) and kv_read(COMPLETED, logdir, index):
        print(f'#{index} already completed, skipping... '
              '(use the --worker.do_if_complete flag to override)')
        return False

    signal.alarm(job_timeout)
    to_log = routine(index=index, logdir=str(worker_logs))

    assert not COMPLETED in to_log, f"no '{COMPLETED}' key allowed in returned data"
    for k, v in to_log.items():
        kv_log(k, v, index=index)

    # add a COMPLETED entry for this set of logs
    kv_log(COMPLETED, 1, index=index)
    with open(Path(logdir) / 'cmds.txt', 'a') as f:
        f.write(' '.join(sys.argv) + '\n')

    return True


@param('worker.main_import')
def main(main_import):
    module = importlib.import_module(main_import)
    make_config(quiet=True)

    routine = module.main

    status = do_index(routine=routine)

if __name__ == '__main__':
    make_config(quiet=True)
    main()
