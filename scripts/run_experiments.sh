#!/usr/bin/sh

EXPERIMENTS_PATH="configs/full_experiments"
EXPERIMENT_FILES="ucf-crime_anomaly_train_vitl14_no-reduce ucf-crime_video_train_vitl14_192 ucf-crime_video_train_vitl14_384 ucf-crime_video_train_vitl14_576 ucf-crime_video_train_vitl14_no-reduce xd-violence_anomaly_train_vitl14_no-reduce xd-violence_video_train_vitl14_192 xd-violence_video_train_vitl14_384 xd-violence_video_train_vitl14_576 xd-violence_video_train_vitl14_no-reduce"
# EXPERIMENT_NAMES
# set -- "weighted-sampler_anomaly_no-reduce_CE-label-smoothing_dropout-0-3" "weighted-sampler_video_reduce-192_CE-label-smoothing_dropout-0-3" "weighted-sampler_video_reduce-384_CE-label-smoothing_dropout-0-3" "weighted-sampler_video_reduce-576_CE-label-smoothing_dropout-0-3" "weighted-sampler_video_reduce-no-reduce_CE-label-smoothing_dropout-0-3" "weighted-sampler_anomaly_no-reduce_CE-label-smoothing_dropout-0-3" "weighted-sampler_video_reduce-192_CE-label-smoothing_dropout-0-3" "weighted-sampler_video_reduce-384_CE-label-smoothing_dropout-0-3" "weighted-sampler_video_reduce-576_CE-label-smoothing_dropout-0-3" "weighted-sampler_video_reduce-no-reduce_CE-label-smoothing_dropout-0-3"
set -- "no-weighted-sampler_anomaly_no-reduce_CE-no-label-smoothing_no-dropout" "no-weighted-sampler_video_reduce-192_CE-no-label-smoothing_no-dropout" "no-weighted-sampler_video_reduce-384_CE-no-label-smoothing_no-dropout" "no-weighted-sampler_video_reduce-576_CE-no-label-smoothing_no-dropout" "no-weighted-sampler_video_reduce-no-reduce_CE-no-label-smoothing_no-dropout" "no-weighted-sampler_anomaly_no-reduce_CE-no-label-smoothing_no-dropout" "no-weighted-sampler_video_reduce-192_CE-no-label-smoothing_no-dropout" "no-weighted-sampler_video_reduce-384_CE-no-label-smoothing_no-dropout" "no-weighted-sampler_video_reduce-576_CE-no-label-smoothing_no-dropout" "no-weighted-sampler_video_reduce-no-reduce_CE-no-label-smoothing_no-dropout"
for experiment in $EXPERIMENT_FILES; do
    echo "Running experiment '$1' with config '$EXPERIMENTS_PATH/$experiment.yaml'\n"
    echo "Running experiment '$1' with config '$EXPERIMENTS_PATH/$experiment.yaml'\n" > experiments_log.txt
    python train.py --gpu 0 -c "$EXPERIMENTS_PATH/$experiment.yaml" --custom-name $1
    shift
done
