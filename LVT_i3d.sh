python main.py --dataset_name LV --device 0 --model_name AED_T --feature_pretrain_model i3d --feature_modal combine --max_seqlen 5 --feature_size 2048 --Lambda 1_10 --ano_class 14 --sample_size 4 --max_epoch 300 --lr 0.00005
#python main.py --dataset_name LAD2000 --device 0 --model_name AED_T --feature_pretrain_model i3d --feature_modal rgb --max_seqlen 5 --feature_size 1024 --Lambda 1_10
#python main.py --dataset_name LAD2000 --device 0 --model_name AED_T --feature_pretrain_model i3d --feature_modal flow --max_seqlen 5 --feature_size 1024 --Lambda 1_10
