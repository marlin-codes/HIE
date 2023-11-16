#!/usr/bin/env bash
model=HGCN
for use_att in 3
do
  for dim in 64
  do
    for hyp_ireg in 0 hir_tangent hire_tangent
    do
      for dropout in 0.5
      do
        for weight_decay in 5e-4
        do
          for dataset in cora
          do
            python train.py \
            --task nc \
            --dataset ${dataset} \
            --model ${model} \
            --manifold PoincareBall \
            --lr 0.01 \
            --cuda 0 \
            --act relu \
            --bias 1 \
            --log-freq 100 \
            --num-layers 2 \
            --patience 500 \
            --use-att ${use_att} \
            --weight-decay ${weight_decay} \
            --dropout ${dropout} \
            --dim ${dim} \
            --hyp_ireg ${hyp_ireg} \
            --ireg_lambda 1.0
          done
        done
      done
    done
  done
done
