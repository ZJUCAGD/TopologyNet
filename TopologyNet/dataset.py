import torch.utils.data as data
import torch
import numpy as np
from persim import PersImage

class ToyDataset(data.Dataset):
    def __init__(self):
        self.dataset = []

        path = '../dataset/toy_data_0.2.npy'
        points = np.load(path, allow_pickle=True)

        points = np.array(points)
        print(points.shape)

        self.dataset = points

    def __getitem__(self, index):
        points = self.dataset[index]
        points = torch.tensor(points).float()
        return points

    def __len__(self):
        return len(self.dataset)

class PIModelNetDataset(data.Dataset):
    def __init__(self,
                 npoints=1024,
                 mode='train'):

        self.dataset = []

        pim = PersImage(spread=1e-2, pixels=[50, 50], verbose=False)

        path = '../dataset/modelNet40_sample.npy'
        points = np.load(path, allow_pickle=True)

        points_avail = []
        pds = []
        pdsh2 = []

        if mode == 'train':
            for i in range(3000):
                h1_file_name = '1_' + str(i) + '.txt'
                h2_file_name = '2_' + str(i) + '.txt'
                pd = np.loadtxt('../PD_modelnet/' + h1_file_name)
                pdh2 = np.loadtxt('../PD_modelnet/' + h2_file_name)
                if len(pd) == 0 or len(pdh2) == 0:
                    continue
                if len(pd.shape) == 1:
                    pd = np.reshape(pd, (1, -1))
                if len(pdh2.shape) == 1:
                    pdh2 = np.reshape(pdh2, (1, -1))
                pds.append(pd)
                pdsh2.append(pdh2)
                points_avail.append(points[i])
        else:
            for i in range(3000, len(points)):
                h1_file_name = '1_' + str(i) + '.txt'
                h2_file_name = '2_' + str(i) + '.txt'
                pd = np.loadtxt('../PD_modelnet/' + h1_file_name)
                pdh2 = np.loadtxt('../PD_modelnet/' + h2_file_name)
                if len(pd) == 0 or len(pdh2) == 0:
                    continue
                if len(pd.shape) == 1:
                    pd = np.reshape(pd, (1, -1))
                if len(pdh2.shape) == 1:
                    pdh2 = np.reshape(pdh2, (1, -1))
                pds.append(pd)
                pdsh2.append(pdh2)
                points_avail.append(points[i])

        pis = []
        for i in range(len(pds)):
            pis.append(pim.transform([pds[i]]))
        pis = np.array(pis).astype(np.float32)
        pis = pis / pis.max()

        pish2 = []
        for i in range(len(pdsh2)):
            pish2.append(pim.transform([pdsh2[i]]))
        pish2 = np.array(pish2).astype(np.float32)
        pish2 = pish2 / pish2.max()

        for i in range(len(points_avail)):
            self.dataset.append((points_avail[i], pis[i], pish2[i]))

        self.npoints = npoints
        print(len(self.dataset))

    def __getitem__(self, index):
        points, pi, pih2 = self.dataset[index]
        points = torch.tensor(points).float()
        pi = torch.tensor(pi.copy()).float()
        pi = pi.view(2500,)

        pih2 = torch.tensor(pih2.copy()).float()
        pih2 = pih2.view(2500,)
        return points, pi, pih2

    def __len__(self):
        return len(self.dataset)