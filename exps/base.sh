export NCCL_IB_DISABLE=1
./tools/run_dist_launch.sh 8 ./configs/r50_deformable_detr.sh   \
   --num_queries 1000 --epochs 50 --enc_layers 6 --dec_layers 6 \
   --with_box_refine  --lr_drop 40 --batch_size 1 --aps 0       \
   --output_dir base
bash exps/base_test.sh 49
bash exps/base_test.sh 48
bash exps/base_test.sh 47
