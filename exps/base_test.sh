export NCCL_IB_DISABLE=1
python3 testing.py  --num_gpus 8                          \
 --coco_path /home/zhenganlin/june/CrowdHuman/annotations \
 --num_queries 1000 --output_dir output/eval_dump         \
 --batch_size 1 --start_epoch $1 --end_epoch $[$1+1]      \
 --with_box_refine --output_dir base --aps 0
python3 demo_opt.py base $1