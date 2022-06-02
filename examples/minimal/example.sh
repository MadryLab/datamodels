#!/bin/bash

set -e
tmp_dir=/tmp/$RANDOM
mkdir $tmp_dir

echo "Logging in $tmp_dir"
python -m datamodels.training.initialize_store \
    --logging.logdir=$tmp_dir \
    --logging.spec=examples/minimal/spec.json

python -m datamodels.training.worker \
    --worker.index=0 \
    --worker.main_import=examples.minimal.train_minimal \
    --worker.logdir=$tmp_dir \
    --trainer.multiple=2

python -m examples.minimal.check \
    --checking.logdir=$tmp_dir \
    --trainer.multiple=2
