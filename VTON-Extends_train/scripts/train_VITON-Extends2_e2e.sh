python -m torch.distributed.launch --nproc_per_node=8 --master_port=7129 train_VITON-Extends2_e2e.py --name VITON-Extends2_e2e   \
--VITON-Extends2_warp_checkpoint 'checkpoints/VITON-Extends2_stage1/VITON-Extends2_warp_epoch_201.pth'  \
--VITON-Extends_warp_checkpoint 'checkpoints/VITON-Extends_e2e/VITON-Extends_warp_epoch_101.pth' --VITON-Extends_gen_checkpoint 'checkpoints/VITON-Extends_e2e/VITON-Extends_gen_epoch_101.pth'  \
--resize_or_crop None --verbose --tf_log --batchSize 4 --num_gpus 8 --label_nc 14 --launcher pytorch










