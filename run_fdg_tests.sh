#!/bin/bash

python Test_Setup_Main.py 8 -1 fdg 1 Models_Medical_10_8 Output_MainTestSetup_10 10 8 5 > dc_hb_fdg_bs_8_lr_-1_epochs_10_vgg_5_logs 2>&1 &
sleep 5s
python Test_Setup_Main_HB.py 8 -1 fdg 1 Models_Medical_10_8 Output_MainTestSetup_10 10 5 > hb_fdg_bs_8_lr_-1_epochs_10_vgg_5_logs 2>&1 &
sleep 5s
python Test_Setup_Main.py 16 -1 fdg 1 Models_Medical_10_16 Output_MainTestSetup_10 10 8 5 > dc_hb_fdg_bs_16_lr_-1_epochs_10_vgg_5_logs 2>&1 &
sleep 5s
python Test_Setup_Main_HB.py 16 -1 fdg 1 Models_Medical_10_16 Output_MainTestSetup_10 10 5 > hb_fdg_bs_16_lr_-1_epochs_10_vgg_5_logs 2>&1 &
