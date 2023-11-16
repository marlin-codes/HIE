from __future__ import division
from __future__ import print_function

import datetime
import json
import logging
import os
import pickle
import time

import numpy as np
import optimizers
import torch
from config import parser
from models.base_models import NCModel
from utils.data_utils import load_data, mkdirs
from utils.train_utils import get_dir_name, format_metrics


def train(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if int(args.double_precision):
        torch.set_default_dtype(torch.float64)
    if int(args.cuda) >= 0:
        torch.cuda.manual_seed(args.seed)
    args.device = 'cuda:' + str(args.cuda) if int(args.cuda) >= 0 else 'cpu'
    args.patience = args.epochs if not args.patience else  int(args.patience)
    logging.getLogger().setLevel(logging.INFO)
    if args.save_log:
        if not args.save_dir:
            dt = datetime.datetime.now()
            date = f"{dt.year}_{dt.month}_{dt.day}"
            models_dir = mkdirs(os.path.join('logs', args.task, date))  # logs/task/date/run/
            save_dir = get_dir_name(models_dir)
        else:
            save_dir = args.save_dir
        logging.basicConfig(level=logging.INFO,
                            handlers=[
                                logging.FileHandler(os.path.join(save_dir, 'log.txt')),
                                logging.StreamHandler()
                            ])

    logging.info(f'Using: {args.device}')
    logging.info("Using seed {}.".format(args.seed))

    # Load data
    data = load_data(args, os.path.join('./data', args.dataset))
    args.n_nodes, args.feat_dim = data['features'].shape
    if args.task == 'nc':
        Model = NCModel
        args.n_classes = int(data['labels'].max() + 1)
        logging.info(f'Num classes: {args.n_classes}')
    else:
        args.nb_false_edges = len(data['train_edges_false'])
        args.nb_edges = len(data['train_edges'])
        if args.task == 'lp':
            Model = LPModel
        else:
            raise ValueError(f'Invalid task type {args.task}')

    if not args.lr_reduce_freq:
        args.lr_reduce_freq = args.epochs

    # Model and optimizer
    model = Model(args)
    logging.info(str(model))
    optimizer = getattr(optimizers, args.optimizer)(params=model.parameters(), lr=args.lr,
                                                    weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=int(args.lr_reduce_freq),
        gamma=float(args.gamma)
    )
    tot_params = sum([np.prod(p.size()) for p in model.parameters()])
    logging.info(f"Total number of parameters: {tot_params}")
    if args.cuda is not None and int(args.cuda) >= 0:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.cuda)
        model = model.to(args.device)
        for x, val in data.items():
            if torch.is_tensor(data[x]):
                data[x] = data[x].to(args.device)
    # Train model
    t_total = time.time()
    counter = 0
    best_val_metrics = model.init_metric_dict()
    best_test_metrics = None
    best_emb = None
    for epoch in range(args.epochs):
        t = time.time()
        model.train()
        optimizer.zero_grad()
        embeddings = model.encode(data['features'], data['adj_train_norm'])
        train_metrics = model.compute_metrics(embeddings, data, 'train')
        train_metrics['loss'].backward()
        if args.grad_clip is not None:
            max_norm = float(args.grad_clip)
            all_params = list(model.parameters())
            for param in all_params:
                torch.nn.utils.clip_grad_norm_(param, max_norm)
        optimizer.step()
        lr_scheduler.step()
        if (epoch + 1) % args.log_freq == 0:
            logging.info(" ".join(['Epoch: {:04d}'.format(epoch + 1),
                                   'lr: {}'.format(lr_scheduler.get_last_lr()[0]),
                                   format_metrics(train_metrics, 'train'),
                                   'time: {:.4f}s'.format(time.time() - t)
                                   ]))
        if (epoch + 1) % args.eval_freq == 0:
            model.eval()
            embeddings = model.encode(data['features'], data['adj_train_norm'])
            val_metrics = model.compute_metrics(embeddings, data, 'val')
            if (epoch + 1) % args.log_freq == 0:
                logging.info(" ".join(['Epoch: {:04d}'.format(epoch + 1), format_metrics(val_metrics, 'val')]))
            if model.has_improved(best_val_metrics, val_metrics):
                best_test_metrics = model.compute_metrics(embeddings, data, 'test')
                best_emb = embeddings.cpu()
                # if args.save:
                #     np.save(os.path.join(save_dir, 'embeddings.npy'), best_emb.detach().numpy())
                best_val_metrics = val_metrics
                counter = 0
            else:
                counter += 1
                if counter == args.patience and epoch > args.min_epochs:
                    logging.info("Early stopping")
                    break

    logging.info("Optimization Finished!")
    logging.info("Total time elapsed: {:.4f}s".format(time.time() - t_total))
    if not best_test_metrics:
        model.eval()
        best_emb = model.encode(data['features'], data['adj_train_norm'])
        best_test_metrics = model.compute_metrics(best_emb, data, 'test')
    logging.info(" ".join(["Val set results:", format_metrics(best_val_metrics, 'val')]))
    logging.info(" ".join(["Test set results:", format_metrics(best_test_metrics, 'test')]))

    best_result = [best_test_metrics['acc'], best_test_metrics['f1']] if args.task == 'nc' else [best_test_metrics['roc'], best_test_metrics['ap']]
    results.append(best_result)
    seeds.append(args.seed)



if __name__ == '__main__':
    results = []
    seeds = []
    args = parser.parse_args()
    print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
    print('>>>>>>>START TRAINING<<<<<<<<<')
    print('\t dataset:', args.dataset)
    print('\t dimension:', args.dim)
    print('\t regularization type:', args.hyp_ireg)
    print('\t regularization parameter', args.ireg_lambda)
    print('>>>>>>>START TRAINING<<<<<<<<<')
    train(args)
    print('======='*20)
    logging.info(args)
    print('======='*20)



    # if args.save_acc_roc:
    #     results_dir = mkdirs('./results/{}/{}/'.format(args.dirname, args.dataset))
    #     f = open(results_dir + '{}_{}_{}.txt'.format(args.dr, args.dim, args.use_att), 'w')
    # for i in range(args.runs):
    #     args.seed = 1234 + i
    #     # try:
    #     train(args)
    #     # except:
    #     #     continue
    #     logging.info(args)
    #     args.seed = args.seed - i
    # for r, s in zip(results, seeds):
    #     logging.info("Results of seed {}: {:.1f}, {:.1f}".format(s, r[0]*100, r[1]*100))
    #     if args.save_acc_roc:
    #         f.write("Results of seed {}: {:.1f}, {:.1f}\n".format(s, r[0]*100, r[1]*100))
    # if args.runs > 1:
    #     results.remove(max(results))
    #     results.remove(min(results))
    #
    #     auc_mean = np.mean(np.array(results)[:, 0])
    #     auc_std = np.std(np.array(results)[:, 0])
    #
    #     ap_mean = np.mean(np.array(results)[:, 1])
    #     ap_std = np.std(np.array(results)[:, 1])
    #
    #     logging.info("Average Results of all runs: {:.1f}+{:.1f}\t {:.1f}+{:.1f}".format(auc_mean*100, auc_std*100, ap_mean*100, ap_std*100))
    #     if args.save_acc_roc:
    #         f.write("Average Results of all runs: {:.1f}+{:.1f}\t {:.1f}+{:.1f}\n".format(auc_mean*100, auc_std*100, ap_mean*100, ap_std*100))
    #         f.close()
            # args = parser.parse_args()
            # train(args)
