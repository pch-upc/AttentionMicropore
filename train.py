import os
import argparse
import itertools

import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch

from models import GeneratorAtL2H, GeneratorAtH2L, DiscriminatorL, DiscriminatorH
from utils import ReplayBuffer
from utils import LambdaLR
from utils import Logger
from datasets import ImageDataset

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=1, help='starting epoch')
    parser.add_argument('--n_epochs', type=int, default=40, help='number of epochs of training')
    parser.add_argument('--batch_size', type=int, default=4, help='size of the batches')
    parser.add_argument('--dataroot', type=str, default='./data/WZ', help='root directory of the dataset')
    parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate')
    parser.add_argument('--decay_epoch', type=int, default=20, help='epoch to start linearly decaying the learning rate to 0')
    parser.add_argument('--size_LR', type=int, default=64, help='size of the data crop (squared assumed)')
    parser.add_argument('--size_HR', type=int, default=256, help='size of the data crop (squared assumed)')
    parser.add_argument('--input_nc', type=int, default=1, help='number of channels of input data')
    parser.add_argument('--cuda', "--test_action",default='True',action='store_true', help='use GPU computation')
    parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
    opt = parser.parse_args()
    print(opt)

    model_dir = './save_models'

    if torch.cuda.is_available() and not opt.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    ###### Definition of variables ######
    # Networks
    netG_A2B = GeneratorAtL2H(opt.input_nc)
    netG_B2A = GeneratorAtH2L(opt.input_nc)
    netD_A = DiscriminatorL(opt.input_nc)
    netD_B = DiscriminatorH(opt.input_nc)

    if opt.cuda:
        netG_A2B.cuda()
        netG_B2A.cuda()
        netD_A.cuda()
        netD_B.cuda()

    # Lossess
    criterion_GAN = torch.nn.MSELoss()
    criterion_cycle = torch.nn.L1Loss()
    criterion_identity = torch.nn.L1Loss()

    # Optimizers & LR schedulers
    optimizer_G = torch.optim.Adam(itertools.chain(netG_A2B.parameters(), netG_B2A.parameters()), lr=opt.lr, betas=(0.5, 0.999))
    optimizer_D_A = torch.optim.Adam(netD_A.parameters(), lr=opt.lr, betas=(0.5, 0.999))
    optimizer_D_B = torch.optim.Adam(netD_B.parameters(), lr=opt.lr, betas=(0.5, 0.999))

    lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)
    lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(optimizer_D_A, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)
    lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(optimizer_D_B, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)

    # Inputs & targets memory allocation
    Tensor = torch.cuda.FloatTensor if opt.cuda else torch.Tensor
    input_A = Tensor(opt.batch_size, opt.input_nc, opt.size_LR, opt.size_LR)
    input_B = Tensor(opt.batch_size, opt.input_nc, opt.size_HR, opt.size_HR)
    target_real = Variable(Tensor(opt.batch_size, 1).fill_(1.0), requires_grad=False)
    target_fake = Variable(Tensor(opt.batch_size, 1).fill_(0.0), requires_grad=False)

    fake_A_buffer = ReplayBuffer()
    fake_B_buffer = ReplayBuffer()

    # Dataset loader
    transform = [transforms.ToTensor(),
                    transforms.Normalize((0.5,), (0.5,)) ]
    dataloader = DataLoader(ImageDataset(opt.dataroot, transform=transform, unaligned=False), 
                            batch_size=opt.batch_size, shuffle=True, num_workers=opt.n_cpu)

    # Loss plot
    logger = Logger(opt.n_epochs, len(dataloader))
    ###################################

    ###### Training ######
    for epoch in range(opt.epoch, opt.n_epochs):
        for i, batch in enumerate(dataloader):
            # Set model input
            real_A = Variable(input_A.copy_(batch['A']))
            real_B = Variable(input_B.copy_(batch['B']))

            ###### Generators A2B and B2A ######
            optimizer_G.zero_grad()

            # Identity loss
            # G_A2B(B) should equal B if real B is fed
            down_B = torch.nn.functional.interpolate(input=real_B,scale_factor=0.25,mode='bicubic')
            same_B = netG_A2B(down_B)[0]
            loss_identity_B = criterion_identity(same_B, real_B)*5.0
            # G_B2A(A) should equal A if real A is fed
            up_A = torch.nn.functional.interpolate(input=real_A,scale_factor=4,mode='bicubic')
            same_A = netG_B2A(up_A)[0]
            loss_identity_A = criterion_identity(same_A, real_A)*5.0

            # GAN loss
            fake_B = netG_A2B(real_A)[0]
            pred_fake = netD_B(fake_B)
            loss_GAN_A2B = criterion_GAN(pred_fake, target_real)

            fake_A = netG_B2A(real_B)[0]
            pred_fake = netD_A(fake_A)
            loss_GAN_B2A = criterion_GAN(pred_fake, target_real)

            # Cycle loss
            recovered_A = netG_B2A(fake_B)[0]
            loss_cycle_ABA = criterion_cycle(recovered_A, real_A)*10.0

            recovered_B = netG_A2B(fake_A)[0]
            loss_cycle_BAB = criterion_cycle(recovered_B, real_B)*10.0

            # Total loss
            loss_G = loss_identity_A + loss_identity_B + loss_GAN_A2B + loss_GAN_B2A + loss_cycle_ABA + loss_cycle_BAB
            loss_G.backward()
            
            optimizer_G.step()
            ###################################

            ###### Discriminator A ######
            optimizer_D_A.zero_grad()

            # Real loss
            pred_real = netD_A(real_A)
            loss_D_real = criterion_GAN(pred_real, target_real)

            # Fake loss
            fake_A = fake_A_buffer.push_and_pop(fake_A)
            pred_fake = netD_A(fake_A.detach())
            loss_D_fake = criterion_GAN(pred_fake, target_fake)

            # Total loss
            loss_D_A = (loss_D_real + loss_D_fake)*0.5
            loss_D_A.backward()

            optimizer_D_A.step()
            ###################################

            ###### Discriminator B ######
            optimizer_D_B.zero_grad()

            # Real loss
            pred_real = netD_B(real_B)
            loss_D_real = criterion_GAN(pred_real, target_real)
            
            # Fake loss
            fake_B = fake_B_buffer.push_and_pop(fake_B)
            pred_fake = netD_B(fake_B.detach())
            loss_D_fake = criterion_GAN(pred_fake, target_fake)

            # Total loss
            loss_D_B = (loss_D_real + loss_D_fake)*0.5
            loss_D_B.backward()

            optimizer_D_B.step()
            ###################################

            # Progress report (http://localhost:8097)
            logger.log(losses={'loss_G': loss_G.item(), 'loss_G_identity': (loss_identity_A.item() + loss_identity_B.item()), 'loss_G_GAN': (loss_GAN_A2B.item() + loss_GAN_B2A.item()),
                        'loss_G_cycle': (loss_cycle_ABA.item() + loss_cycle_BAB.item()), 'loss_D': (loss_D_A.item() + loss_D_B.item())}, 
                        images={'real_A': real_A, 'real_B': real_B, 'fake_A': fake_A, 'fake_B': fake_B})

            # f = open("training_curve.csv", "a")
            # f.write('[%d/%d][%d/%d] loss_G: %.4f loss_G_identity: %.4f loss_G_GAN: %.4f loss_G_cycle: %.4f loss_D: %.4f'
            #     % (epoch, opt.n_epochs, i+1, len(dataloader),
            #         loss_G.item(),(loss_identity_A.item() + loss_identity_B.item()),(loss_GAN_A2B.item() + loss_GAN_B2A.item()),
            #         (loss_cycle_ABA.item() + loss_cycle_BAB.item()),(loss_D_A.item() + loss_D_B.item())))                 
            # f.write('\n')
            # f.close()
            
        # Update learning rates
        lr_scheduler_G.step()
        lr_scheduler_D_A.step()
        lr_scheduler_D_B.step()

        # Save models checkpoints
        torch.save(netG_A2B.state_dict(), os.path.join(model_dir, 'netG_A2B_%d.pth'%epoch))
        torch.save(netG_B2A.state_dict(), os.path.join(model_dir, 'netG_B2A_%d.pth'%epoch))
        torch.save(netD_A.state_dict(), os.path.join(model_dir, 'netD_A_%d.pth'%epoch))
        torch.save(netD_B.state_dict(), os.path.join(model_dir, 'netD_B_%d.pth'%epoch))
    ###################################
if __name__ == '__main__':
    main()