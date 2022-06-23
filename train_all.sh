#!/bin/sh

# define any variables if needed:
PROJECT_NAME_1='Representation-learning-teacher'
PROJECT_NAME_2='Representation-learning-student'




############################################################## TEACHER ON HUMANS ##############################################################


# # resnet50:
# python train_teacher.py --model resnet50 --train_dir ./data/human_train.csv --valid_dir ./data/human_valid.csv --save_model_path ./checkpoints/thsa/teacher/resnet50/ \
# --name resnet50 --project $PROJECT_NAME_1 --group teacher_on_humans 

# # swin-tiny:
# python train_teacher.py --model swin-tiny --train_dir ./data/human_train.csv --valid_dir ./data/human_valid.csv --save_model_path ./checkpoints/thsa/teacher/swin-tiny/ \
# --name resnet50 --project $PROJECT_NAME_1 --group teacher_on_humans  

# # vit-small:
# python train_teacher.py --model vit-small --train_dir ./data/human_train.csv --valid_dir ./data/human_valid.csv --save_model_path ./checkpoints/thsa/teacher/vit-small/ \
# --name resnet50 --project $PROJECT_NAME_1 --group teacher_on_humans  

# now student


# # resnet50:
# python train_student.py --model resnet50 --train_dir ./data/animal_train.csv --valid_dir ./data/animal_valid.csv --save_model_path ./checkpoints/thsa/student/resnet50/ \
# --teacher_weights ./checkpoints/thsa/teacher/resnet50/best-ckpt.pt --name resnet50 --project $PROJECT_NAME_2 --group student_on_animals 

# # swin-tiny:
# python train_student.py --model swin-tiny --train_dir ./data/animal_train.csv --valid_dir ./data/animal_valid.csv --save_model_path ./checkpoints/thsa/student/swin-tiny/ \
# --teacher_weights ./checkpoints/thsa/teacher/swin-tiny/best-ckpt.pt --name swin-tiny --project $PROJECT_NAME_2 --group student_on_animals --volume_loss yes

# # vit-small:
# python train_student.py --model vit-small --train_dir ./data/animal_train.csv --valid_dir ./data/animal_valid.csv --save_model_path ./checkpoints/thsa/student/vit-small/ \
# --teacher_weights ./checkpoints/thsa/teacher/vit-small/best-ckpt.pt --name vit-small --project $PROJECT_NAME_2 --group student_on_animals --volume_loss yes


############################################################## TEACHER ON ANIMALS ##############################################################


# resnet50:
python train_teacher.py --model resnet50 --train_dir ./data/animal_train.csv --valid_dir ./data/animal_valid.csv --save_model_path ./checkpoints/tash/teacher/resnet50/ \
--name resnet50 --project $PROJECT_NAME_1 --group teacher_on_animals

# swin-tiny:
python train_teacher.py --model swin-tiny --train_dir ./data/animal_train.csv --valid_dir ./data/animal_valid.csv --save_model_path ./checkpoints/tash/teacher/swin-tiny/ \
--name resnet50 --project $PROJECT_NAME_1 --group teacher_on_animals

# vit-small:
python train_teacher.py --model vit-small --train_dir ./data/animal_train.csv --valid_dir ./data/animal_valid.csv --save_model_path ./checkpoints/tash/teacher/vit-small/ \
--name resnet50 --project $PROJECT_NAME_1 --group teacher_on_animals


# now student


# resnet50:
python train_student.py --model resnet50 --train_dir ./data/human_train.csv --valid_dir ./data/human_valid.csv --save_model_path ./checkpoints/tash/student/resnet50/ \
--teacher_weights ./checkpoints/tash/teacher/resnet50/best-ckpt.pt --name resnet50 --project $PROJECT_NAME_2 --group student_on_humans 

# swin-tiny:
python train_student.py --model swin-tiny --train_dir ./data/human_train.csv --valid_dir ./data/human_valid.csv --save_model_path ./checkpoints/tash/student/swin-tiny/ \
--teacher_weights ./checkpoints/tash/teacher/swin-tiny/best-ckpt.pt --name swin-tiny --project $PROJECT_NAME_2 --group student_on_humans --volume_loss yes

# vit-small:
python train_student.py --model vit-small --train_dir ./data/human_train.csv --valid_dir ./data/human_valid.csv --save_model_path ./checkpoints/tash/student/vit-small/ \
--teacher_weights ./checkpoints/tash/teacher/vit-small/best-ckpt.pt --name vit-small --project $PROJECT_NAME_2 --group student_on_humans --volume_loss yes