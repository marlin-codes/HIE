#!/usr/bin/env bash
for use_att in 1
do
  for dim in 8 64 256
  do
    for hyp_ireg in 0
    do
      for dropout in 0.6
      do
        for weight_decay in 1e-4
        do
          for dataset in citeseer
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
            --num-layers 2 \
            --patience 500 \
            --use-att ${use_att} \
            --weight-decay ${weight_decay} \
            --dropout ${dropout} \
            --dim ${dim} \
            --hyp_ireg ${hyp_ireg} \
            --ireg_lambda 0 \
            --runs 1
          done
        done
      done
    done
  done
done
