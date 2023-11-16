root=0
leaf=2
link = []
def tree(level_1, leaf=leaf):
    level_2 = []
    for i in level_1:
        for j in range(i + leaf-1 + leaf * (i-1), i + leaf + leaf * i):
            level_2.append(j)
            # print(i-1, j-1)
            link.append((i-1,j-1))
            # link.append((j-1,i-1))
    return level_2


level_1=list(range(1, leaf))
print(len(link))
# print(link)
level_2 = tree(level_1)
print(len(link))
level_3 = tree(level_2)
# print(link)
print(len(link))
level_4 = tree(level_3)
# print(link)
print(len(link))
level_5 = tree(level_4)
# print(link)
print(len(link))
# exit()
level_6 = tree(level_5)
print(len(link))
level_7 = tree(level_6)
print(len(link))
# level_8 = tree(level_7)
# print(len(link))

def assortative_rate(G):
    graph_rate = []
    G = G.to_undirected()
    for center in G.nodes():
        node_rate = []
        y = G.nodes[center]['label']
        for neighbor in G.neighbors(center):
            # print(G.nodes[neighbor]['label'], G.nodes[center]['label'])
            if int(G.nodes[center]['label'])==int(G.nodes[neighbor]['label']):
                node_rate.append(1)
            else:
                node_rate.append(0)
        graph_rate.append(np.mean(node_rate))
    return np.mean(graph_rate)

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import networkx as nx
    import numpy as np
    import scipy.sparse as sp
    G = nx.DiGraph()
    G.add_edges_from(link)

    # g0 = (1, 0)
    # g1 = (1, 1)
    # g2 = (1, 2)
    # g3 = (1, 3)
    # g0 = (0.00,   0.50)
    # g1 = (0.00,   0.45)
    # g2 = (0.00,   0.40)
    # g3 = (0.00,   0.35)
    # g4 = (0.00,   0.30)
    # g5 = (0.00,   0.25)
    # g6 = (0.00,   0.20)
    # g7 = (0.00,   0.15)
    g0 = (0, 1)
    g1 = (1,   0.5)
    g2 = (1.5,   0.5)
    g3 = (2,   0.5)
    g4 = (2.5,   0.5)
    g5 = (3,   0.5)
    g6 = (3.5,   0.5)
    g7 = (4,   0.5)

    def addlabel(node):
        if node<=0:
            G.nodes[node]['label'] = 0
            G.nodes[node]['feat'] = list(g0[0] * np.random.randn(32) + g0[1])
        elif node<=3:
            G.nodes[node]['label'] = 1
            G.nodes[node]['feat'] = list(g1[0] * np.random.randn(32) + g1[1])
        elif node<=12:
            G.nodes[node]['label'] = 2
            G.nodes[node]['feat'] = list(g2[0] * np.random.randn(32) + g2[1])
        elif node <= 39:
            G.nodes[node]['label'] = 3
            G.nodes[node]['feat'] = list(g3[0] * np.random.randn(32) + g3[1])
        elif node <= 120:
            G.nodes[node]['label'] = 4
            G.nodes[node]['feat'] = list(g4[0] * np.random.randn(32) + g4[1])
        elif node <= 363:
            G.nodes[node]['label'] = 5
            G.nodes[node]['feat'] = list(g5[0] * np.random.randn(32) + g5[1])
        else:
            G.nodes[node]['label'] = 6
            G.nodes[node]['feat'] = list(g6[0] * np.random.randn(32) + g6[1])
        # else:
        #     G.nodes[node]['label'] = 7
        #     G.nodes[node]['feat'] = list(g7[0] * np.random.randn(32) + g7[1])

    edge_index = np.array(G.edges())
    for node in G.nodes():
        addlabel(node)

    label = list(dict(G.nodes(data='label')).values())
    feat = np.array(list(dict(G.nodes(data='feat')).values()))
    print('number of nodes:', G.number_of_nodes())
    print('number of edges:', G.number_of_edges())
    print('number of labels:', len(label))
    print('number of feat:', feat.shape)
    print('assortative rate:', assortative_rate(G))
    np.savetxt('./data/treel/treel.edges.csv',edge_index, fmt='%d',delimiter=',')
    np.save('./data/treel/treel.labels.npy', label)
    sp.save_npz('./data/treel/treel.feats', sp.csc_matrix(feat))
    print('TREE_L edges/labels/feats has been updated')
