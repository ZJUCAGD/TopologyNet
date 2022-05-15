import argparse
import os
import random
import torch.optim as optim
import torch.utils.data
from TopologyNet.dataset import PIModelNetDataset
from TopologyNet.model import TopologyNet
import numpy as np
from matplotlib import pyplot as plt
import torch.nn.functional as F
from tqdm import tqdm
from tensorboardX import SummaryWriter

def visualization(pred, points, name):
    fig = plt.figure(figsize=[16, 8])
    ax1 = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122)

    ax1.scatter(points[0, :], points[1, :], points[2, :])
    ax2.imshow(pred.reshape(50, 50))

    if not os.path.exists('../output'):
        os.makedirs('../output')

    fig.savefig('../output/{}.jpg'.format(name))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--batchSize', type=int, default=8, help='input batch size')
    parser.add_argument(
        '--workers', type=int, help='number of data loading workers', default=4)
    parser.add_argument(
        '--nepoch', type=int, default=10, help='number of epochs to train for')
    parser.add_argument('--outf', type=str, default='pis', help='output folder')
    parser.add_argument('--model', type=str, default='', help='model path')
    parser.add_argument('--phase', type=str, default='train', help='train or test')

    opt = parser.parse_args()
    print(opt)

    blue = lambda x: '\033[94m' + x + '\033[0m'

    opt.manualSeed = 8002
    print("Random Seed: ", opt.manualSeed)
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)

    # trainset = PIModelNetDataset(mode='train')
    testset = PIModelNetDataset(mode='test')

    # trainloader = torch.utils.data.DataLoader(trainset, batch_size=opt.batchSize, shuffle=True, num_workers=int(opt.workers))
    testloader = torch.utils.data.DataLoader(testset, batch_size=opt.batchSize, shuffle=False, num_workers=int(opt.workers))

    # print(len(trainset), len(testset))
    try:
        os.makedirs(opt.outf)
    except OSError:
        pass

    model = TopologyNet()

    if opt.model != '':
        model.load_state_dict(torch.load(opt.model))

    optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    model.cuda()

    writer = SummaryWriter('runs/scalar')

    if opt.phase == 'train':
        num_batch = len(trainset) / opt.batchSize
        for epoch in range(opt.nepoch):
            scheduler.step()
            for i, data in enumerate(trainloader):
                points, pi, pih2 = data

                points = points.transpose(2, 1)
                points, pi = points.cuda(), pi.cuda()

                pih2 = pih2.cuda()
                optimizer.zero_grad()
                model = model.train()

                predh1, predh2 = model(points)
                lossh1 = F.mse_loss(predh1, pi)
                lossh2 = F.mse_loss(predh2, pih2)
                loss = lossh1 + lossh2

                writer.add_scalar('train_loss', loss, global_step=epoch*len(trainloader)+i)
                writer.add_scalar('train_loss_h1', lossh1, global_step=epoch*len(trainloader)+i)
                writer.add_scalar('train_loss_h2', lossh2, global_step=epoch*len(trainloader)+i)

                loss.backward()
                optimizer.step()
                print('[%d: %d/%d] train mse loss: %f' % (epoch, i, num_batch, loss.item()))

                if i % 50 == 0:
                    j, data = next(enumerate(testloader, 0))
                    points, pi, pih2 = data
                    points = points.transpose(2, 1)
                    points, pi = points.cuda(), pi.cuda()
                    pih2 = pih2.cuda()
                    model = model.eval()
                    predh1, predh2 = model(points)

                    lossh1 = F.mse_loss(predh1, pi)
                    lossh2 = F.mse_loss(predh2, pih2)

                    loss = lossh1 + lossh2

                    writer.add_scalar('test_loss', loss, global_step=epoch * len(trainloader) + i)
                    writer.add_scalar('test_loss_h1', lossh1, global_step=epoch * len(trainloader) + i)
                    writer.add_scalar('test_loss_h2', lossh2, global_step=epoch * len(trainloader) + i)

                    print('[%d: %d/%d] %s loss: %f' % (epoch, i, num_batch, blue('test'), loss.item()))

        torch.save(model.state_dict(), '%s/total_pi_model_%d.pth' % (opt.outf, epoch))

    elif opt.phase == 'test':
        total_testset = 0
        loss = 0
        lh1 = 0
        lh2 = 0
        total_loss1 = 0
        total_loss2 = 0

        total_norm_h1 = 0
        total_norm_h2 = 0

        h1_norm_list = []
        h2_norm_list = []

        for i, data in tqdm(enumerate(testloader)):
            points, pi, pih2 = data

            points = points.transpose(2, 1)
            points, pi = points.cuda(), pi.cuda()
            pih2 = pih2.cuda()
            model = model.eval()
            print(points.shape)
            predh1, predh2 = model(points)

            lossh1 = F.mse_loss(predh1, pi)

            lossh2 = F.mse_loss(predh2, pih2)

            lossh1 = lossh1.item()
            lossh2 = lossh2.item()

            loss = lossh1 + lossh2

            total_loss1 += lossh1
            total_loss2 += lossh2

            lh1 = lossh1
            lh2 = lossh2

            points, pi, pih2 = data

            total_testset += 1

            print("final mse {}".format(loss))
            print("h1 {}".format(lh1))
            print("h2 {}".format(lh2))

            total_norm_h1 += lh1/pi.max()
            total_norm_h2 += lh2/pih2.max()

            print("normalizeh1 {}".format(lh1/pi.max()))
            print("normalizeh2 {}".format(lh2/pih2.max()))

            h1_norm_list.append(lh1/pi.max())
            h2_norm_list.append(lh2/pih2.max())

        print("final mse {}".format(total_loss1/total_testset+total_loss2/total_testset))
        print("h1 {}".format(total_loss1/total_testset))
        print("h2 {}".format(total_loss2/total_testset))
        print("total_norm_h1 {}".format(total_norm_h1/total_testset))
        print("total_norm_h2 {}".format(total_norm_h2/total_testset))

        print('avg h1 max', total_loss1/total_norm_h1)
        print("avg h2 max", total_loss2/total_norm_h2)
        print(np.mean(h1_norm_list))
        print(np.mean(h2_norm_list))
        print(np.std(h1_norm_list, ddof=1))
        print(np.std(h2_norm_list, ddof=1))

    elif opt.phase == 'plot':
        for i, data in tqdm(enumerate(testloader)):
            points, pi, pih2 = data
            points = points.transpose(2, 1)
            points = points.cuda()
            model = model.eval()
            predh1, predh2 = model(points)

            points = points.cpu().detach().numpy()
            pred = predh1.cpu().detach().numpy()

            pred = pred[1]
            pi = pi[1]
            points = points[0]

            visualization(pred, points, 'pred')
            visualization(pi, points, 'gt')

            break


