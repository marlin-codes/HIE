#!/usr/bin/env bash
for use_att in 0
do
  for dim in 8
  do
    for dr in 0 1 6
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
            --use-att ${use_att} \
            --weight-decay ${weight_decay} \
            --dropout ${dropout} \
            --dim ${dim} \
            --dr ${dr} \
            --runs 10 \
            --save_acc_roc 1
          done
        done
      done
    done
  done
done
