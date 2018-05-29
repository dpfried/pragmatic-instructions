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

num_candidates=20
inference_type="beam"
sample_alpha=1.0

output_dir="expts/rational_speaker/${corpus}/test_sentence-segmented_contextual_ens=10_${inference_type}${sample_alpha}=${num_candidates}_actions=5_tune-acc-end-state+bleu"
mkdir -p $output_dir
log_file=${output_dir}/log.out

python -u -m scone.rational_speaker \
    --dynet_mem 3000 \
    --corpus $corpus \
    --follower_dirs expts/follower/${corpus}/factored-multihead-contextual_feed-actions_d=${f_dropout}_dim=${f_dim}_att-dim=${f_att}/{1..10}/ \
    --speaker_dirs expts/speaker/${corpus}/d=${s_dropout}_dim=${s_dim}/{1..10}/ \
    --inference_type ${inference_type} \
    --sample_alpha ${sample_alpha} \
    --num_candidates $num_candidates \
    --verbose \
    --sentence_segmented \
    --test_decode \
    --prediction_output_dir $output_dir \
    --tuning_metrics acc_end_state bleu \
    | tee $log_file
