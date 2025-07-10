#!/bin/bash
for BASE_CONFIG in   'base_chexpert' 'base_domainnet' 'base_retina' 'base_living17' 'base_entity30' 'base_camelyon' 'base_icam' 'base_density'
do
    echo $BASE_CONFIG
    python classification/train.py experiment=$BASE_CONFIG model.encoder_name=efficientnet_b0 trainer.use_focal_loss=True
    python classification/train.py  experiment=$BASE_CONFIG model.encoder_name=mobilenetv2_100 trainer.use_focal_loss=True
    python classification/train.py experiment=$BASE_CONFIG model.encoder_name=vit_base_patch16_224 trainer.use_focal_loss=True trainer.lr=1e-5
    python classification/train.py experiment=$BASE_CONFIG model.encoder_name=convnext_tiny trainer.use_focal_loss=TrueA trainer.lr=1e-4
    python classification/train.py experiment=$BASE_CONFIG model.encoder_name=resnet50 trainer.use_focal_loss=True
    python classification/train.py experiment=$BASE_CONFIG model.encoder_name=resnet18 trainer.use_focal_loss=True
    for LABEL_SMOOTHING in 0 0.05
    do
        for ER in 0 0.1
        do
            echo $LABEL_SMOOTHING $ER
            python classification/train.py experiment=$BASE_CONFIG model.encoder_name=efficientnet_b0 trainer.entropy_regularisation=$ER trainer.label_smoothing=$LABEL_SMOOTHING
            python classification/train.py experiment=$BASE_CONFIG model.encoder_name=mobilenetv2_100 trainer.entropy_regularisation=$ER trainer.label_smoothing=$LABEL_SMOOTHING
            python classification/train.py experiment=$BASE_CONFIG model.encoder_name=vit_base_patch16_224 trainer.entropy_regularisation=$ER trainer.lr=1e-5 trainer.label_smoothing=$LABEL_SMOOTHING
            python classification/train.py experiment=$BASE_CONFIG model.encoder_name=convnext_tiny trainer.entropy_regularisation=$ER trainer.lr=1e-4 trainer.label_smoothing=$LABEL_SMOOTHING
            python classification/train.py experiment=$BASE_CONFIG model.encoder_name=resnet50 trainer.entropy_regularisation=$ER trainer.label_smoothing=$LABEL_SMOOTHING
            python classification/train.py experiment=$BASE_CONFIG model.encoder_name=resnet18 trainer.entropy_regularisation=$ER trainer.label_smoothing=$LABEL_SMOOTHING
        done
    done
done