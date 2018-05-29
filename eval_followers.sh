#!/bin/bash
corpus=$1

if [ "$corpus" == "alchemy" ]; then
        f_dropout=0.1
        f_dim=50
        f_att=50
        s_dropout=0.3
        s_dim=100
elif [ "$corpus" == "scene" ]; then
        f_dropout=0.1
        f_dim=100
        f_att=100
        s_dropout=0.3
        s_dim=100
elif [ "$corpus" == "tangrams" ]; then
        f_dropout=0.3
        f_dim=50
        f_att=100
        s_dropout=0.3
        s_dim=50
else 
        echo "invalid corpus $corpus"
        exit 1
fi

num_candidates=40

output_dir="expts/rational_follower/${corpus}/contextual-ens=${num_models}_beam=${num_candidates}-test"
mkdir $output_dir
log_file=${output_dir}/log.out

python -u -m scone.rational_follower \
    --dynet_mem 8000 \
    --corpus $corpus \
    --follower_dirs expts/follower/${corpus}/factored-multihead-contextual_feed-actions_d=${f_dropout}_dim=${f_dim}_att-dim=${f_att}/{1..10}/ \
    --speaker_dirs expts/speaker/${corpus}/d=${s_dropout}_dim=${s_dim}/{1..10}/ \
    --inference_type beam \
    --num_candidates ${num_candidates} \
    --verbose \
    --test_decode \
    --prediction_output_dir $output_dir \
    | tee $log_file
