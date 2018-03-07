#!/bin/bash

python Test_Setup_Main_HB.py 50 -1 cifar10 > hb_cifar10_bs_50_lr_-1_logs 2>&1 &
python Test_Setup_Main_HB.py 100 -1 cifar10 > hb_cifar10_bs_100_lr_-1_logs 2>&1 &
python Test_Setup_Main_HB.py 200 -1 cifar10 > hb_cifar10_bs_200_lr_-1_logs 2>&1 &
python Test_Setup_Main_HB.py 500 -1 cifar10 > hb_cifar10_bs_500_lr_-1_logs 2>&1 &


#python Test_Setup_Main.py 50 0 cifar10 > dc_hb_cifar10_bs_50_lr_0_logs 2>&1 &
#python Test_Setup_Main.py 100 0 cifar10 > dc_hb_cifar10_bs_100_lr_0_logs 2>&1 &
#python Test_Setup_Main.py 200 0 cifar10 > dc_hb_cifar10_bs_200_lr_0_logs 2>&1 &
#python Test_Setup_Main.py 500 0 cifar10 > dc_hb_cifar10_bs_500_lr_0_logs 2>&1 &

