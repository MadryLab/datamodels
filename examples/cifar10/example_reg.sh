#!/bin/bash

echo "> Setting up regression..."

# Change to OUT_DIR from the training script
tmp_dir=/tmp/27501

python -m datamodels.regression.write_dataset \
             --cfg.data_dir $tmp_dir \
             --cfg.out_path "$tmp_dir/reg_data.beton" \
             --cfg.y_name margins \
             --cfg.x_name masks
echo "> regression data prepared!"

echo "> starting regression..."
python -m datamodels.regression.compute_datamodels \
    -C examples/cifar10/regression_config.yaml \
    --data.data_path "$tmp_dir/reg_data.beton" \
    --cfg.out_dir "$tmp_dir/reg_results"
echo "> regression DONE!"
echo "> Datamodels stored in: $tmp_dir/reg_results/datamodels.pt" 
