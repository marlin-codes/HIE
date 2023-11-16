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
level_2 = tree(level_1)
level_3 = tree(level_2)
level_4 = tree(level_3)
level_5 = tree(level_4)

level_6 = tree(level_5)
level_7 = tree(level_6)

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import networkx as nx
    import numpy as np
    import scipy.sparse as sp
    G = nx.DiGraph()
    G.add_edges_from(link)

    g0 = (1, 0)
    g1 = (1, 1)
    g2 = (1, 2)
    g3 = (1, 3)
    G.nodes[0]['label'] = 0
    G.nodes[1]['label'] = 1
    G.nodes[2]['label'] = 2
    G.nodes[3]['label'] = 3

    def addlabel(node, label):
        successors = G.successors(node)
        if G.successors:
            G.nodes[node]['label']=label
            if label==1:
                G.nodes[node]['feat'] = list(g1[0]*np.random.randn(32)+g1[1])

            if label==2:
                G.nodes[node]['feat'] = list(g2[0]*np.random.randn(32)+g2[1])

            if label==3:
                G.nodes[node]['feat'] = list(g3[0]*np.random.randn(32)+g3[1])

            for neighbor in successors:
                addlabel(neighbor, label)
    for i, node in enumerate([1,2,3]):
        addlabel(node, node)
    G.nodes[0]['label'] = 0
    G.nodes[0]['feat'] = list(g0[0]*np.random.randn(32)+g0[1])

    label = list(dict(G.nodes(data='label')).values())
    edge_index = np.array(G.edges())
    feat = np.array(list(dict(G.nodes(data='feat')).values()))
    print('number of nodes:', G.number_of_nodes())
    print('number of edges:', G.number_of_edges())
    print('number of labels:', len(label))
    print('number of feat:', feat.shape)

    np.savetxt('./data/tree/tree.edges.csv',edge_index, fmt='%d',delimiter=',')
    np.save('./data/tree/tree.labels.npy',label)
    sp.save_npz('./data/tree/tree.feats', sp.csc_matrix(feat))
    print('tree edges/labels/feats has been updated')
