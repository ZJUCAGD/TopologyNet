import numpy as np
from ripser import Rips
from tqdm import tqdm
import os

input = np.load('dataset/modelNet40_sample.npy', allow_pickle=True)

for index in tqdm(range(0, len(input))):
    rips = Rips(maxdim=2)
    data = input[index]
    dgm = rips.fit_transform(data)
    np.savetxt('./PD_modelnet/0_' + str(index) + '.txt', np.array(dgm[0]), fmt='%s')
    np.savetxt('./PD_modelnet/1_' + str(index) + '.txt', np.array(dgm[1]), fmt='%s')
    np.savetxt('./PD_modelnet/2_' + str(index) + '.txt', np.array(dgm[2]), fmt='%s')
