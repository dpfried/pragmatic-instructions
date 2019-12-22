#!/bin/bash
corpus=$1
seed=$2
action_layer=$3
latent_beam_size=$4

if [[ -z $action_layer ]]; then
  action_layer="factored_multihead_contextual"
fi

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

if [[ -z $latent_beam_size ]]; then
  latent_beam_size=10
fi

out_suffix="${action_layer}_feed-actions_d=${f_dropout}_dim=${f_dim}_att-dim=${f_att}_bs=${latent_beam_size}"
out_dir="expts/latent_follower/${corpus}/${out_suffix}"

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
        --action_layer $action_layer \
        --feed_actions_to_decoder \
        --train_epochs 25 \
        --save_dir $model_dir \
        --latent_actions \
        --latent_beam_size $latent_beam_size \
        --train_original \
        --train_original_no_add_actions \
        | tee $log_file
