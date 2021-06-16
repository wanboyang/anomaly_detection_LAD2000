CUDA_VISIBLE_DEVICE=0 python main.py --dataset_name UCF_Crime --device 0 --model_name AED_T --feature_pretrain_model i3d --feature_modal combine --max_seqlen 10 --feature_size 2048 --Lambda 1_10
