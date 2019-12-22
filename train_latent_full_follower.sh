#!/bin/bash
corpus=$1
seed=$2
action_layer=$3
latent_beam_size=$4

exploration_method=$5
beta=$6

max_transitions_per_train_instance=$7

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

if [[ -z $exploration_method ]]; then
  exploration_method="beam"
fi

if [[ -z $beta ]]; then
  beta="1.0"
fi

if [[ -z $max_transitions_per_train_instance ]]; then
  max_transitions_per_train_instance=5
fi


out_suffix="${action_layer}_feed-actions_d=${f_dropout}_dim=${f_dim}_att-dim=${f_att}_bs=${latent_beam_size}_exp=${exploration_method}_beta=${beta}_max-trans=${max_transitions_per_train_instance}"
out_dir="expts/latent_full_follower/${corpus}/${out_suffix}"

log_file="${out_dir}/${seed}.out"
model_dir="${out_dir}/${seed}/"

mkdir -p $out_dir
mkdir -p $model_dir

python -u -m scone.follower \
        --dynet_seed $seed \
        --dynet_mem 2000 \
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
        --exploration_method $exploration_method \
        --latent_update_beta $beta \
        --max_transitions_per_train_instance $max_transitions_per_train_instance \
        | tee $log_file
