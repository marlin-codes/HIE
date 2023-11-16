#!/usr/bin/env bash
for dim in 8 64 256
  do
    for hyp_ireg in hir_tangent
    do
        for dataset in disease_nc
        do
        python train.py \
        --task nc \
        --dataset ${dataset} \
        --model HNN \
        --dropout 0.1 \
        --manifold PoincareBall \
        --weight-decay 5e-4 \
        --lr 0.01 \
        --num-layers 3 \
        --patience 400 \
        --log-freq 100 \
        --act relu \
        --cuda 0 \
        --bias 1 \
        --dim ${dim} \
        --hyp_ireg ${hyp_ireg} \
        --runs 1 \
        --ireg_lambda 0.01
    done
  done
done
