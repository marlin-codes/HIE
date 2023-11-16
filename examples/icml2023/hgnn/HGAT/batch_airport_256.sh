#!/usr/bin/env bash
model=HGCN
for use_att in 3
do
  for dim in 256
  do
    for hyp_ireg in 0 hir_tangent hire_tangent
    do
        for dataset in airport
        do
        python train.py \
        --task nc \
        --dataset ${dataset} \
        --model ${model} \
        --dropout 0 \
        --weight-decay 0 \
        --manifold PoincareBall \
        --normalize-feats 0 \
        --lr 0.01 \
        --num-layers 4 \
        --patience 500 \
        --log-freq 200 \
        --act relu \
        --cuda 0 \
        --bias 1 \
        --dim ${dim} \
        --hyp_ireg ${hyp_ireg} \
        --use-att ${use_att} \
        --ireg_lambda 0.01
       done
    done
  done
done
