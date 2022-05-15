import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
from TopologyNet.dataset import ToyDataset
import torch
from TopologyNet.model import TopologyNet
import time

def visualization(pred, points, name):
    fig = plt.figure(figsize=[16, 8])
    ax1 = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122)

    ax1.scatter(points[0, :], points[1, :], points[2, :])
    ax2.imshow(pred.reshape(50, 50))

    if not os.path.exists('../output'):
        os.makedirs('../output')

    fig.savefig('../output/{}.jpg'.format(name))

def cycle(n_sample=1000, noise=0, graph=False):
    r = 1
    theta = np.random.uniform(0, 2 * np.pi, n_sample)
    x = r * np.cos(theta) + np.random.normal(0, noise, n_sample)
    y = r * np.sin(theta) + np.random.normal(0, noise, n_sample)
    z = np.random.normal(0, noise, n_sample)

    if graph:
        fig = plt.figure()
        ax1 = fig.add_subplot(111, projection='3d')
        ax1.scatter(x, y, z, c='g', marker='o')
        ax1.set_xlim([-1, 1])
        ax1.set_ylim([-1, 1])
        ax1.set_zlim([-1, 1])
        ax1.set_xlabel('x axis')
        ax1.set_ylabel('y axis')
        ax1.set_zlabel('z axis')
        plt.show()

    rotation_angle = np.random.uniform(0, 2 * np.pi, 1)[0]
    cos_theta = np.cos(rotation_angle)
    sin_theta = np.sin(rotation_angle)
    rotation_matrix = np.array([[cos_theta, sin_theta, 0],
                                [0, 0, 1],
                                [-sin_theta, cos_theta, 0]])

    return np.dot(np.column_stack([x, y, z]), rotation_matrix)


def double_cycle1(n_sample=1000, noise=0, graph=False):
    r1 = 0.6
    theta1 = np.random.uniform(0, 2 * np.pi, n_sample // 2)
    x1 = r1 * np.cos(theta1)
    y1 = r1 * np.sin(theta1)

    r2 = 0.2
    theta2 = np.random.uniform(0, 2 * np.pi, n_sample // 2)
    x2 = r2 * np.cos(theta2)
    y2 = r2 * np.sin(theta2)

    x = np.hstack([x1, x2]) + np.random.normal(0, noise, n_sample)
    y = np.hstack([y1, y2]) + np.random.normal(0, noise, n_sample)
    z = np.random.normal(0, noise, n_sample)

    if graph:
        fig = plt.figure()
        ax1 = fig.add_subplot(111, projection='3d')
        ax1.scatter(x, y, z, c='g', marker='o')
        ax1.set_xlim([-1, 1])
        ax1.set_ylim([-1, 1])
        ax1.set_zlim([-1, 1])
        ax1.set_xlabel('x axis')
        ax1.set_ylabel('y axis')
        ax1.set_zlabel('z axis')
        plt.show()

    rotation_angle = np.random.uniform(0, 2 * np.pi, 1)[0]
    cos_theta = np.cos(rotation_angle)
    sin_theta = np.sin(rotation_angle)
    rotation_matrix = np.array([[cos_theta, sin_theta, 0],
                                [0, 0, 1],
                                [-sin_theta, cos_theta, 0]])

    return np.dot(np.column_stack([x, y, z]), rotation_matrix)


def double_cycle2(n_sample=1000, noise=0, graph=False):
    r1 = 0.3
    theta1 = np.random.uniform(0, 2 * np.pi, n_sample // 2)
    x1 = r1 * np.cos(theta1) + 0.5
    y1 = r1 * np.sin(theta1) + 0.5

    r2 = 0.3
    theta2 = np.random.uniform(0, 2 * np.pi, n_sample // 2)
    x2 = r2 * np.cos(theta2) - 0.5
    y2 = r2 * np.sin(theta2) - 0.5

    x = np.hstack([x1, x2]) + np.random.normal(0, noise, n_sample)
    y = np.hstack([y1, y2]) + np.random.normal(0, noise, n_sample)
    z = np.random.normal(0, noise, n_sample)

    if graph:
        fig = plt.figure()
        ax1 = fig.add_subplot(111, projection='3d')
        ax1.scatter(x, y, z, c='g', marker='o')
        ax1.set_xlim([-1, 1])
        ax1.set_ylim([-1, 1])
        ax1.set_zlim([-1, 1])
        ax1.set_xlabel('x axis')
        ax1.set_ylabel('y axis')
        ax1.set_zlabel('z axis')
        plt.show()

    rotation_angle = np.random.uniform(0, 2 * np.pi, 1)[0]
    cos_theta = np.cos(rotation_angle)
    sin_theta = np.sin(rotation_angle)
    rotation_matrix = np.array([[cos_theta, sin_theta, 0],
                                [0, 0, 1],
                                [-sin_theta, cos_theta, 0]])

    return np.dot(np.column_stack([x, y, z]), rotation_matrix)



def rand1(n_sample=1000, noise=0, graph=False):
    result = np.random.uniform(-0.5, 0.5, size=(n_sample, 3)) + np.random.normal(0, noise, (n_sample, 3))
    x = result[:, 0]
    y = result[:, 1]
    z = result[:, 2]

    if graph:
        fig = plt.figure()
        ax1 = fig.add_subplot(111, projection='3d')
        ax1.scatter(x, y, z, c='g', marker='o')
        ax1.set_xlim([-1, 1])
        ax1.set_ylim([-1, 1])
        ax1.set_zlim([-1, 1])
        ax1.set_xlabel('x axis')
        ax1.set_ylabel('y axis')
        ax1.set_zlabel('z axis')
        plt.show()

    rotation_angle = np.random.uniform(0, 2 * np.pi, 1)[0]
    cos_theta = np.cos(rotation_angle)
    sin_theta = np.sin(rotation_angle)
    rotation_matrix = np.array([[cos_theta, sin_theta, 0],
                                [0, 0, 1],
                                [-sin_theta, cos_theta, 0]])

    return np.dot(np.column_stack([x, y, z]), rotation_matrix)

def sphere(n_sample=1000, noise=0, graph=False):
    theta = np.random.uniform(0, 180, size=n_sample)
    phi = np.random.uniform(0, 360, size=n_sample)

    r = 0.5

    x = r * np.sin(theta) * np.cos(phi) + np.random.normal(0, noise, (n_sample,))
    y = r * np.sin(theta) * np.sin(phi) + np.random.normal(0, noise, (n_sample,))
    z = r * np.cos(theta) + np.random.normal(0, noise, (n_sample,))

    if graph:
        fig = plt.figure()
        ax1 = fig.add_subplot(111, projection='3d')
        ax1.scatter(x, y, z, c='g', marker='o')
        ax1.set_xlim([-1, 1])
        ax1.set_ylim([-1, 1])
        ax1.set_zlim([-1, 1])
        ax1.set_xlabel('x axis')
        ax1.set_ylabel('y axis')
        ax1.set_zlabel('z axis')
        plt.show()

    rotation_angle = np.random.uniform(0, 2 * np.pi, 1)[0]
    cos_theta = np.cos(rotation_angle)
    sin_theta = np.sin(rotation_angle)
    rotation_matrix = np.array([[cos_theta, sin_theta, 0],
                                [0, 0, 1],
                                [-sin_theta, cos_theta, 0]])

    return np.dot(np.column_stack([x, y, z]), rotation_matrix)

def rand2(n_sample=1000, noise=0, graph=False):
    result1 = np.random.uniform(-0.5, 0, size=(n_sample // 2, 2))
    result2 = np.random.uniform(0, 0.5, size=(n_sample // 2, 2))
    result = np.vstack([result1, result2]) + np.random.normal(0, noise, (n_sample, 2))

    x = result[:, 0]
    y = result[:, 1]
    z = np.random.normal(0, noise, n_sample)

    if graph:
        fig = plt.figure()
        ax1 = fig.add_subplot(111, projection='3d')
        ax1.scatter(x, y, z, c='g', marker='o')
        ax1.set_xlim([-1, 1])
        ax1.set_ylim([-1, 1])
        ax1.set_zlim([-1, 1])
        ax1.set_xlabel('x axis')
        ax1.set_ylabel('y axis')
        ax1.set_zlabel('z axis')
        plt.show()


    return np.column_stack([x, y, z])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='generate', help='generate/evaluate')
    parser.add_argument('--model', type=str, default='pis/total_pi_model_9.pth', help='root of model')

    opt = parser.parse_args()
    print(opt)

    if opt.mode == 'generate':
        noises = [0.2]
        for noise in noises:
            data = []
            for i in range(100):
                data.append(cycle(n_sample=512, noise=noise, graph=False))

            for i in range(100):
                data.append(double_cycle1(n_sample=512, noise=noise, graph=False))

            for i in range(100):
                data.append(double_cycle2(n_sample=512, noise=noise, graph=False))

            for i in range(100):
                data.append(rand1(n_sample=512, noise=noise, graph=False))

            for i in range(100):
                data.append(sphere(n_sample=512, noise=noise, graph=False))

            data = np.array(data)
            print(data.shape)
            np.save('../dataset/toy_data_{}.npy'.format(noise), data)


    elif opt.mode == 'evaluate':
        toyset = ToyDataset()
        toyloader = torch.utils.data.DataLoader(toyset, batch_size=32, shuffle=False,
                                                 num_workers=4)

        print(len(toyset))
        model = TopologyNet()
        model.cuda()

        visualize = False

        if opt.model != '':
            model.load_state_dict(torch.load(opt.model))

        result_pi = []

        t1 = time.time()
        for i, points in enumerate(toyloader):
            points = points.transpose(2, 1)
            points = points.cuda()
            model = model.eval()
            predh1, predh2 = model(points)

            points = points.cpu().detach().numpy()
            pred = predh1.cpu().detach().numpy()

            if visualize:
                pred = pred[0]
                points = points[0]

                visualization(pred, points, '{}'.format(i))
                exit()

            else:
                for pi in pred:
                    result_pi.append(pi)

        t2 = time.time()
        print(t2-t1)

        result_pi = np.array(result_pi)

        print(result_pi.shape)

        np.save('toy_pi_0.2.npy', result_pi)




