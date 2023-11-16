#!/usr/bin/env bash
model=HGCN
for dim in 8
do
  for hyp_ireg in 0 hir_tangent hire_tangent
  do
    for dropout in 0.1
    do
      for weight_deacy in 5e-3
      do
        for n_heads in 1
        do
          for use_att in 3
          do
          python train.py \
          --task nc \
          --dataset pubmed \
          --model ${model} \
          --dropout ${dropout} \
          --weight-decay ${weight_deacy} \
          --manifold PoincareBall \
          --lr 0.01 \
          --cuda 0 \
          --log 100 \
          --patience 500 \
          --num-layers 3 \
          --n-heads ${n_heads} \
          --act relu \
          --bias 1 \
          --use-att ${use_att} \
          --dim ${dim} \
          --hyp_ireg ${hyp_ireg} \
          --ireg_lambda 0.1
        done
        done
      done
    done
  done
done