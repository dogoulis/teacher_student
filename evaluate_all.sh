#!/bin/sh

# define any variables if needed:


PROJECT_NAME_1='EVALUATE_ON_ANIMALS'
PROJECT_NAME_2='EVALUATE_ON_HUMANS'

########################## EVALUATE ON ANIMALS ##########################

# # resnet50
# python evaluation.py --model resnet50 --test_dir ./data/animal_test.csv --batch_size 32 --student_weights ./checkpoints/tash/student/resnet50/best-ckpt.pt \
# --project_name $PROJECT_NAME_1 --name student --group resnet50
# # baseline
# python evaluation.py --model resnet50 --test_dir ./data/animal_test.csv --batch_size 32 \
# --project_name $PROJECT_NAME_1 --name baseline --group resnet50

# # swin-tiny 
# python evaluation.py --model swin-tiny --test_dir ./data/animal_test.csv --batch_size 32 --student_weights ./checkpoints/tash/student/swin-tiny/best-ckpt.pt \
# --project_name $PROJECT_NAME_1 --name student --group swin-tiny 
# # baseline
# python evaluation.py --model swin-tiny --batch_size 32 --test_dir ./data/animal_test.csv \
# --project_name $PROJECT_NAME_1 --name baseline --group swin-tiny 

# # vit-small
# python evaluation.py --model vit-small --test_dir ./data/animal_test.csv --batch_size 32 --student_weights ./checkpoints/tash/student/vit-small/best-ckpt.pt \
# --project_name $PROJECT_NAME_1 --name student --group vit-small 
# # baseline
# python evaluation.py --model vit-small --batch_size 32 --test_dir ./data/animal_test.csv \
# --project_name $PROJECT_NAME_1 --name baseline --group vit-small 

########################## EVALUATE ON HUMANS ##########################

# # resnet50
# python evaluation.py --model resnet50 --test_dir ./data/human_test.csv --batch_size 32 --student_weights ./checkpoints/thsa/student/resnet50/best-ckpt.pt \
# --project_name $PROJECT_NAME_2 --name student --group resnet50 --device 'cuda:1'
# # baseline
# python evaluation.py --model resnet50 --test_dir ./data/human_test.csv --batch_size 32 \
# --project_name $PROJECT_NAME_2 --name baseline --group resnet50 --device 'cuda:1'

# swin-tiny 
python evaluation.py --model swin-tiny --test_dir ./data/human_test.csv --batch_size 32 --student_weights ./checkpoints/thsa/student/swin-tiny/best-ckpt.pt \
--project_name $PROJECT_NAME_2 --name student --group swin-tiny 
# baseline
python evaluation.py --model swin-tiny --batch_size 32 --test_dir ./data/human_test.csv \
--project_name $PROJECT_NAME_2 --name baseline --group swin-tiny --device 'cuda:1'

# # vit-small
# python evaluation.py --model vit-small --test_dir ./data/human_test.csv --batch_size 32 --student_weights ./checkpoints/thsa/student/vit-small/best-ckpt.pt \
# --project_name $PROJECT_NAME_2 --name student --group vit-small --device 'cuda:1'
# # baseline
# python evaluation.py --model vit-small --batch_size 32 --test_dir ./data/human_test.csv \
# --project_name $PROJECT_NAME_2 --name baseline --group vit-small --device 'cuda:1'

