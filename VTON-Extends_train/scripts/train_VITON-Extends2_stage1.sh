python -m torch.distributed.launch --nproc_per_node=8 --master_port=4703 train_VITON-Extends2_stage1.py --name VITON-Extends2_stage1  \
--PBAFN_warp_checkpoint 'checkpoints/VITON-Extends_e2e/VITON-Extends_warp_epoch_101.pth' --PBAFN_gen_checkpoint 'checkpoints/VITON-Extends_e2e/VITON-Extends_gen_epoch_101.pth'  \
--lr 0.00003 --niter 100 --niter_decay 100 --resize_or_crop None --verbose --tf_log --batchSize 4 --num_gpus 8 --label_nc 14 --launcher pytorch










