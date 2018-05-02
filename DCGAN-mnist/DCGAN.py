import os
import torch
from torch.autograd import Variable
import torchvision.utils as vutils
from utils import Opt
from dataset import train_loader
from model import netD, netG, weights_init

def train(netD, netG, criterion, optimizerG, optimizerD):
    for epoch in range(Opt.epoch):
        avg_lossD = 0
        avg_lossG = 0
        with open(os.path.join(Opt.root, 'logs.txt', 'a')) as file:
            for i, (data, _) in enumerate(train_loader):
                # Update D network
                mini_batch = data.shape[0]
                # train with real
                input = Variable(data.cuda())   # image input
                real_label = Variable(torch.ones(mini_batch).cuda())
                output = netD(input)
                D_real_loss = criterion(output, real_label)
                # train with fake
                noise = Variable(torch.randn(mini_batch, Opt.nz).view(-1, Opt.nz, 1, 1).cuda())
                fake = netG(noise)
                fake_label = Variable(torch.zeros(mini_batch).cuda())
                output = netD(fake.detach())    # detach to avoid training G on these labels
                G_real_loss = criterion(output, fake_label)
                D_loss = D_real_loss + G_real_loss
                netD.zero_grad()
                D_loss.backward()
                if Opt.which_pc == 0:
                    avg_lossD += D_loss.item()
                else:
                    avg_lossD += D_loss.data[0]
                optimizerD.step()
                # Update G network
                output = netD(fake)
                G_loss = criterion(output, real_label)
                if Opt.which_pc == 0:
                    avg_lossG += G_loss.item()
                else:
                    avg_lossG += G_loss.data[0]
                netG.zero_grad()
                G_loss.backward()
                optimizerG.step()

                print('Epoch [%d/%d], Step [%d/%d], D_loss: %.4f, G_loss: %.4f'
                      % (epoch + 1, Opt.epoch, i + 1, len(train_loader), D_loss.data[0], G_loss.data[0]))
            avg_lossD /= i
            avg_lossG /= i
            print('epoch: ' + str(epoch) + ', G_loss: ' + str(avg_lossG) + ', D_loss: ' + str(avg_lossD))
            file.write('epoch: ' + str(epoch) + ', G_loss: ' + str(avg_lossG) + ', D_loss: ' + str(avg_lossD) + '\n')

        # save generated images
        fixed_pred = netG(fixed_noise)
        vutils.save_image(fixed_pred.data, os.path.join(Opt.results_dir,'img'+str(epoch)+'.png'), nrow=10, scale_each=True)

        if epoch % 200 == 0:
            if Opt.save_model:
                torch.save(netD.state_dict(), os.path.join(Opt.checkpoint_dir, 'netD-01.pt'))
                torch.save(netG.state_dict(), os.path.join(Opt.checkpoint_dir, 'netG-01.pt'))


if __name__ == '__main__':
    fixed_noise = Variable(torch.randn(100, Opt.nz).view(-1, Opt.nz, 1, 1).cuda())

    netG = netG()
    # netG.apply(weights_init)
    netD = netD()
    # netD.apply(weights_init)

    netG.cuda()
    netD.cuda()

    # Loss function
    criterion = torch.nn.BCELoss()

    # Optimizers
    optimizerG = torch.optim.Adam(netG.parameters(), lr=Opt.lr, betas=Opt.betas)
    optimizerD = torch.optim.Adam(netD.parameters(), lr=Opt.lr, betas=Opt.betas)

    train(netD, netG, criterion, optimizerG, optimizerD)