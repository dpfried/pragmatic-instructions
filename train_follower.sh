#!/bin/bash
corpus=$1
seed=$2

if [ "$corpus" == "alchemy" ]; then
        f_dropout=0.1
        f_dim=50
        f_att=50
elif [ "$corpus" == "scene" ]; then
        f_dropout=0.1
        f_dim=100
        f_att=100
elif [ "$corpus" == "tangrams" ]; then
        f_dropout=0.3
        f_dim=50
        f_att=100
else 
        echo "invalid corpus $corpus"
        exit 1
fi

out_suffix="factored-multihead-contextual_feed-actions_d=${f_dropout}_dim=${f_dim}_att-dim=${f_att}"
out_dir="expts/follower/${corpus}/${out_suffix}"

log_file="${out_dir}/${seed}.out"
model_dir="${out_dir}/${seed}/"

mkdir -p $out_dir
mkdir -p $model_dir

python -u -m scone.follower \
        --dynet_seed $seed \
        --random_seed $seed \
        --corpus $corpus \
        --bidi \
        --decode_interval 1 \
        --dropout $f_dropout \
        --verbose \
        --enc_state_dim $f_dim \
        --dec_state_dim $f_dim \
        --embedded_y_dim $f_dim \
        --attention \
        --attention_dim $f_att \
        --action_layer factored_multihead_contextual \
        --feed_actions_to_decoder \
        --train_epochs 25 \
        --save_dir $model_dir \
        | tee $log_file
