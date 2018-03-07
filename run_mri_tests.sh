#!/bin/bash
python Test_Setup_Main.py 8 -2 mri 1 Models_Medical_10_8_densenet Output_MainTestSetup_10_densenet 10 8 4 > dc_hb_mri_bs_8_lr_-2_epochs_10_densenet_4_logs 2>&1 &
sleep 4s
python Test_Setup_Main_HB.py 8 -2 mri 1 Models_Medical_10_8_densenet Output_MainTestSetup_10_densenet 10 4 > hb_mri_bs_8_lr_-2_epochs_10_densenet_4_logs 2>&1 &
sleep 4s
python Test_Setup_Main.py 16 -2 mri 1 Models_Medical_10_16 Output_MainTestSetup_10_densenet 10 8 4 > dc_hb_mri_bs_16_lr_-2_epochs_10_densenet_4_logs 2>&1 &
sleep 4s
python Test_Setup_Main_HB.py 16 -2 mri 1 Models_Medical_10_16 Output_MainTestSetup_10_densenet 10 4 > hb_mri_bs_16_lr_-2_epochs_10_densenet_4_logs 2>&1 &
