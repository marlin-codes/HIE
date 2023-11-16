import os
import pickle as pkl
import time

import networkx as nx


def compute_centrality(args, data):

    if args.centrality == 'hc':
        return None

    centrality_path = args.data_root + f'/centrality/{args.dataset}_{args.task}_centrality.pkl'
    if not os.path.isfile(centrality_path):
        print(f'>> loading {args.dataset} {args.task} {args.centrality} centrality directly')
        with open(centrality_path, 'rb') as f:
            centrality = pkl.load(f)
            assert args.centrality in ['bc', 'cc', 'dc', 'hc']
        return centrality[args.centrality]

    edges = data['adj_train_norm']._indices()
    G = nx.Graph()
    for edge in edges.transpose(1, 0).cpu().numpy():
        G.add_edge(edge[0], edge[1])

    start_time = time.time()
    dc = nx.degree_centrality(G)
    print(f'computing dc takes {time.time() - start_time}s')
    start_time = time.time()
    cc = nx.closeness_centrality(G)
    print(f'computing cc takes {time.time() - start_time}s')
    start_time = time.time()
    bc = nx.betweenness_centrality(G)
    print(f'computing bc takes {time.time() - start_time}s')
    start_time = time.time()
    # ec = nx.eigenvector_centrality(G)
    # print(f'computing ec takes {time.time() - start_time}s')

    dc_value = []
    cc_value = []
    bc_value = []
    ec_value = []

    for node in range(nx.number_of_nodes(G)):
        dc_value.append(dc[node])
        cc_value.append(cc[node])
        bc_value.append(bc[node])
        # ec_value.append(ec[node])
        # kc_value.append(kc[node])

    print('======')
    obj = {'hc': [0, 0]}
    obj['dc'] = [dc_value, dc_value.index(max(dc_value))]
    obj['cc'] = [cc_value, cc_value.index(max(cc_value))]
    obj['bc'] = [bc_value, bc_value.index(max(bc_value))]
    # obj['ec'] = [ec_value, ec_value.index(max(ec_value))]
    print('dc', 'cc', 'bc')
    print(obj['dc'][-1], obj['cc'][-1], obj['bc'][-1])

    with open(centrality_path, 'wb') as f:
        pkl.dump(obj, f)
    print(f'{args.dataset}-{args.task} {args.centrality} info has been saved!')
    return obj[args.centrality]
