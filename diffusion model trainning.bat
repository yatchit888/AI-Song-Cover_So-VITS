@echo off
:start
python train_diff.py -c configs/diffusion.yaml
goto start