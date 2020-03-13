export IMG_HEIGHT=137
export IMG_WIDTH=236
export EPOCHS=60
export TRAIN_BATCH_SIZE=32
export TEST_BATCH_SIZE=8
export NUM_WORKERS=8
export MODEL_MEAN="(0.485, 0.456, 0.406)"
export MODEL_STD="(0.229, 0.224, 0.225)"
export BASE_MODEL="se_resnet50"
export TRAINING_FOLDS_CSV="../input/train_folds.csv"

export TRAINING_FOLDS="(0, 1, 2, 3)"
export VALIDATION_FOLDS="(4, )"
export FOLD=4
python train.py