import numpy as np
from utils.config import Config
from math import sqrt as sqrt
from itertools import product as product

mean = []
for k, f in enumerate(Config["feature_maps"]):
    x,y = np.meshgrid(np.arange(f),np.arange(f))
    x = x.reshape(-1)
    y = y.reshape(-1)
    for i, j in zip(y,x):
        # print(x,y)
        f_k = Config["min_dim"] / Config["steps"][k]
        # 计算网格的中心
        cx = (j + 0.5) / f_k
        cy = (i + 0.5) / f_k

        # 求短边
        s_k =  Config["min_sizes"][k]/Config["min_dim"]
        mean += [cx, cy, s_k, s_k]

        # 求长边
        s_k_prime = sqrt(s_k * (Config["max_sizes"][k]/Config["min_dim"]))
        mean += [cx, cy, s_k_prime, s_k_prime]

        # 获得长方形
        for ar in Config["aspect_ratios"][k]:
            mean += [cx, cy, s_k*sqrt(ar), s_k/sqrt(ar)]
            mean += [cx, cy, s_k/sqrt(ar), s_k*sqrt(ar)]
for i,j in zip(mean,mean):
    print(i==j)