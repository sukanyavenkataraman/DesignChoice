#!/bin/bash
python Test_Setup_Main_HB.py 50 0 svhn > hb_svhn_bs_50_lr_0_logs 2>&1 &
python Test_Setup_Main_HB.py 100 0 svhn > hb_svhn_bs_100_lr_0_logs 2>&1 &
python Test_Setup_Main_HB.py 200 0 svhn > hb_svhn_bs_200_lr_0_logs 2>&1 &
python Test_Setup_Main_HB.py 500 0 svhn > hb_svhn_bs_500_lr_0_logs 2>&1 &

#python Test_Setup_Main.py 50 0 svhn > dc_hb_svhn_bs_50_lr_0_logs 2>&1 &
#python Test_Setup_Main.py 100 0 svhn > dc_hb_svhn_bs_100_lr_0_logs 2>&1 &
#python Test_Setup_Main.py 200 0 svhn > dc_hb_svhn_bs_200_lr_0_logs 2>&1 &
#python Test_Setup_Main.py 500 0 svhn > dc_hb_svhn_bs_500_lr_0_logs 2>&1 &

