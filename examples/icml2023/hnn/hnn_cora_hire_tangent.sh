#!/usr/bin/env bash
for dim in 8 64 256
  do
    for hyp_ireg in hire_tangent
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
            --model HNN \
            --manifold PoincareBall \
            --lr 0.01 \
            --cuda 0 \
            --act relu \
            --bias 1 \
            --log-freq 50 \
            --num-layers 3 \
            --patience 500 \
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
