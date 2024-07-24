# Modified from ArSSR GitHub repo

import data
import torch
import model
import argparse
import time
from torch.utils.tensorboard import SummaryWriter

def train_epoch(model, loader, optimizer, loss_fun, device, epoch, total_epochs):
    """
    Train the model for one epoch.

    Parameters
    ----------
    model : nn.Module
        The model to train.
    loader : DataLoader
        The DataLoader for training data.
    optimizer : torch.optim.Optimizer
        The optimizer for updating the model parameters.
    loss_fun : nn.Module
        The loss function.
    device : torch.device
        The device to run the model on.
    epoch : int
        The current epoch number.
    total_epochs : int
        The total number of epochs for training.

    Returns
    -------
    float
        The average training loss for the epoch.
    """
    model.train()
    total_loss = 0
    for i, (img_lr, xyz_hr, img_hr) in enumerate(loader):
        img_lr = img_lr.unsqueeze(1).to(device).float()  # N×1×h×w×d
        img_hr = img_hr.to(device).float().view(img_lr.size(0), -1).unsqueeze(-1)  # N×K×1
        xyz_hr = xyz_hr.view(img_lr.size(0), -1, 3).to(device).float()  # N×K×3

        optimizer.zero_grad()
        img_pre = model(img_lr, xyz_hr)  # N×K×1
        loss = loss_fun(img_pre, img_hr)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        current_lr = optimizer.state_dict()['param_groups'][0]['lr']
        print(f'(TRAIN) Epoch[{epoch}/{total_epochs}], Step[{i + 1}/{len(loader)}], Lr:{current_lr}, Loss:{loss.item():.10f}')
    
    return total_loss / len(loader)

def validate_epoch(model, loader, loss_fun, device):
    """
    Validate the model for one epoch.

    Parameters
    ----------
    model : nn.Module
        The model to validate.
    loader : DataLoader
        The DataLoader for validation data.
    loss_fun : nn.Module
        The loss function.
    device : torch.device
        The device to run the model on.

    Returns
    -------
    float
        The average validation loss for the epoch.
    """
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for i, (img_lr, xyz_hr, img_hr) in enumerate(loader):
            img_lr = img_lr.unsqueeze(1).to(device).float()  # N×1×h×w×d
            xyz_hr = xyz_hr.view(1, -1, 3).to(device).float()  # N×Q×3 (Q=H×W×D)
            img_hr = img_hr.to(device).float().view(1, -1).unsqueeze(-1)  # N×Q×1

            img_pre = model(img_lr, xyz_hr)  # N×Q×1
            loss = loss_fun(img_pre, img_hr)
            total_loss += loss.item()
    
    return total_loss / len(loader)

def main():
    writer = SummaryWriter('./log')
    parser = argparse.ArgumentParser()

    parser.add_argument('-decoder_depth', type=int, default=8, dest='decoder_depth',
                        help='The depth of the decoder network (default=8).')
    parser.add_argument('-decoder_width', type=int, default=256, dest='decoder_width',
                        help='The width of the decoder network (default=256).')
    parser.add_argument('-feature_dim', type=int, default=128, dest='feature_dim',
                        help='The dimension size of the feature vector (default=128).')
    parser.add_argument('-hr_data_train', type=str, default='./data/train', dest='hr_data_train',
                        help='The file path of HR patches for training.')
    parser.add_argument('-hr_data_val', type=str, default='./data/val', dest='hr_data_val',
                        help='The file path of HR patches for validation.')
    parser.add_argument('-lr', type=float, default=1e-4, dest='lr',
                        help='The initial learning rate.')
    parser.add_argument('-lr_decay_epoch', type=int, default=200, dest='lr_decay_epoch',
                        help='Learning rate multiply by 0.5 per lr_decay_epoch.')
    parser.add_argument('-epoch', type=int, default=100, dest='epoch',
                        help='The total number of epochs for training.')
    parser.add_argument('-summary_epoch', type=int, default=10, dest='summary_epoch',
                        help='The current model will be saved per summary_epoch.')
    parser.add_argument('-bs', type=int, default=20, dest='batch_size',
                        help='The number of LR-HR patch pairs (i.e., N in Equ. 3).')
    parser.add_argument('-ss', type=int, default=8000, dest='sample_size',
                        help='The number of sampled voxel coordinates (i.e., K in Equ. 3).')
    parser.add_argument('-gpu', type=int, default=0, dest='gpu',
                        help='The number of GPU.')

    args = parser.parse_args()

    print('Parameter Settings')
    print('------------File------------')
    print(f'hr_data_train: {args.hr_data_train}')
    print(f'hr_data_val: {args.hr_data_val}')
    print('------------Train-----------')
    print(f'lr: {args.lr}')
    print(f'batch_size_train: {args.batch_size}')
    print(f'sample_size: {args.sample_size}')
    print(f'gpu: {args.gpu}')
    print(f'epochs: {args.epoch}')
    print(f'summary_epoch: {args.summary_epoch}')
    print(f'lr_decay_epoch: {args.lr_decay_epoch}')
    print('------------Model-----------')
    print(f'decoder feature_dim: {args.feature_dim}')
    print(f'decoder depth: {args.decoder_depth}')
    print(f'decoder width: {args.decoder_width}')
    
    time.sleep(5)

    train_loader = data.loader_train(in_path_hr=args.hr_data_train, batch_size=args.batch_size,
                                     sample_size=args.sample_size, is_train=True)
    val_loader = data.loader_train(in_path_hr=args.hr_data_val, batch_size=1,
                                   sample_size=args.sample_size, is_train=False)

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    mrirecon_model = model.mrirecon(feature_dim=args.feature_dim,
                                    decoder_depth=args.decoder_depth // 2,
                                    decoder_width=args.decoder_width).to(device)
    loss_fun = torch.nn.L1Loss()
    optimizer = torch.optim.Adam(params=mrirecon_model.parameters(), lr=args.lr)

    for e in range(args.epoch):
        loss_train = train_epoch(mrirecon_model, train_loader, optimizer, loss_fun, device, e + 1, args.epoch)
        writer.add_scalar('MES_train', loss_train, e + 1)

        loss_val = validate_epoch(mrirecon_model, val_loader, loss_fun, device)
        writer.add_scalar('MES_val', loss_val, e + 1)

        if (e + 1) % args.summary_epoch == 0:
            torch.save(mrirecon_model.state_dict(), f'model3/model_param_{e + 1}.pkl')

        if (e + 1) % args.lr_decay_epoch == 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.5

    writer.flush()

if __name__ == '__main__':
    main()
