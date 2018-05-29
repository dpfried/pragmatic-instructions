#!/bin/bash
corpus=$1
seed=$2

if [ "$corpus" == "alchemy" ]; then
        s_dropout=0.3
        s_dim=100
elif [ "$corpus" == "scene" ]; then
        s_dropout=0.3
        s_dim=100
elif [ "$corpus" == "tangrams" ]; then
        s_dropout=0.3
        s_dim=50
else 
        echo "invalid corpus $corpus"
        exit 1
fi

out_suffix="d=${s_dropout}_dim=${s_dim}"
out_dir="expts/speaker/${corpus}/${out_suffix}"

log_file="${out_dir}/${seed}.out"
model_dir="${out_dir}/${seed}/"

mkdir -p $out_dir
mkdir -p $model_dir


python -u -m scone.speaker \
        --dynet_seed $seed \
        --random_seed $seed \
        --corpus $corpus \
        --bidi \
        --decode_interval 1 \
        --dropout $s_dropout \
        --verbose \
        --enc_state_dim $s_dim \
        --dec_state_dim $s_dim \
        --embedded_y_dim $s_dim \
        --save_dir $model_dir \
        --train_epochs 15 \
        | tee $log_file
