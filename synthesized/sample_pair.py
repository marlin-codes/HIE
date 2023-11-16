
import numpy as np
dataset = 'treel'
dataset2 = 'treel'
treeh_label = np.load(f'./data/{dataset2}/{dataset2}.labels.npy')
# print(treeh_label)

pair = []
label0 = np.where(treeh_label==0)[0]
lable1 = np.where(treeh_label==1)[0]
lable2 = np.where(treeh_label==2)[0]
label3 = np.where(treeh_label==3)[0]

for i in label0:
    for j in lable2:
        pair.append([i,j])
        # print((i,j))
for m in lable1:
    for n in label3:
        pair.append([m,n])
        # print((m,n))
np.random.seed(2022)
idx = list(np.random.choice(len(pair), 1000))
ordered_edges = np.array(pair)[idx]
# print(ordered_edges)

hdo0 = np.loadtxt(f'./results/distance_curv/dist_data_final/{dataset}/{dataset}_8/{dataset}_HDO0.txt')
hdo1 = np.loadtxt(f'./results/distance_curv/dist_data_final/{dataset}/{dataset}_8/{dataset}_HDO1.txt')



a = 0
b = 0
for (x,y) in ordered_edges:
    # print(x, y)
    if hdo0[x]<hdo0[y]:
        a+=1
    if hdo1[x]<hdo1[y]:
        b+=1
print(a,b)





