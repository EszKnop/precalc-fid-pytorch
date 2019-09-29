import argparse
import numpy as np
import torch
import torch.nn.parallel
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms

from inception import InceptionV3

def get_activations_for_dataloader(model, dataloader, cuda=True, verbose=True):
    model.eval()

    pred_arr = list()
    for i, data in enumerate(dataloader, 0):
        if verbose:
            print(f'\rPropagating batch {(i + 1)}', end='', flush=True)
        images = data[0]
        if cuda:
            images = images.cuda()

        pred = model(images)[0]
        pred_arr.append(pred.view(images.size(0), -1))

    result = torch.stack(pred_arr).cpu().data.numpy()

    if verbose:
        print('done. Result size: ', result.size)

    return result
    
def calculate_activation_statistics_for_dataloader(model, dataloader, cuda=False, verbose=False):
    act = get_activations_for_dataloader(model, dataloader, cuda, verbose)
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma

def run():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataroot', type=str)
    parser.add_argument('--imageSize', type=int, default=None)
    parser.add_argument('--batchSize', type=int, default=50)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--normalize', action='store_true')
    parser.add_argument('--outf', type=str, default='fid_stats.npz')

    args = parser.parse_args()

    transforms_collection = []
    if args.imageSize is not None:
        print("Adding resizing")
        transforms_collection = [
            transforms.Resize(args.imageSize),
            transforms.CenterCrop(args.imageSize)
        ]
    transforms_collection += [
        transforms.ToTensor()
    ]
    if args.normalize:
        transforms_collection += [
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]

    dataset = dset.ImageFolder(root=args.dataroot, transform=transforms.Compose(transforms_collection))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batchSize, shuffle=False, num_workers=args.workers)
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
    model = InceptionV3([block_idx]).cuda()
    print("calculte FID stats..", end=" ", flush=True)
    mu, sigma = calculate_activation_statistics_for_dataloader(model, dataloader, cuda=True, verbose=True)
    np.savez_compressed(args.outf, mu=mu, sigma=sigma)
    print("finished")

if __name__ == '__main__':
    run()

