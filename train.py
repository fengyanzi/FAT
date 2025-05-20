import os
import torch.nn as nn
import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import argparse
import random
import numpy as np
from tqdm import tqdm
from models.FAT import FAT
from datetime import datetime
from torchvision import models
import torch.nn.functional as f

class SRDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.HR_dir = os.path.join(root_dir, 'HR')
        self.LR_dir = os.path.join(root_dir, 'LR')
        self.image_filenames = [img for img in os.listdir(self.LR_dir) if
                                img.endswith('.png') or img.endswith('.jpg')]
        self.transform1 = transforms.Compose([
            transforms.Resize(120),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])])
        self.transform2 = transforms.Compose([
            transforms.Resize(480),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])])

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        LR_img_name = os.path.join(self.LR_dir, self.image_filenames[idx])
        HR_img_name = os.path.join(self.HR_dir, self.image_filenames[idx])
        LR_img = Image.open(LR_img_name) #.convert('L')
        HR_img = Image.open(HR_img_name) #.convert('L')
        HR_img = self.transform2(HR_img)
        LR_img = self.transform1(LR_img)
        sample = {'LR_img': LR_img, 'HR_img': HR_img}
        return sample

class PerceptualLoss():
    def contentFunc(self):
        conv_3_3_layer = 14
        cnn = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features
        cnn = cnn.cuda()
        model = nn.Sequential()
        model = model.cuda()
        for i, layer in enumerate(list(cnn)):
            model.add_module(str(i), layer)
            if i == conv_3_3_layer:
                break
        return model

    def __init__(self, loss):
        self.criterion = torch.nn.MSELoss()
        self.contentFunc = self.contentFunc()

    def get_loss(self, fakeIm, realIm):
        f_fake = self.contentFunc.forward(fakeIm)
        f_real = self.contentFunc.forward(realIm)
        f_real_no_grad = f_real.detach()
        loss = self.criterion(f_fake, f_real_no_grad)
        return loss


class VarianceimpFeatureLoss(nn.Module):
    def __init__(self):
        super(VarianceimpFeatureLoss, self).__init__()
        # 定义窗口大小和步幅
        self.window_size0 = 4
        self.stride0 = 4
        self.loss = nn.L1Loss()

    def forward(self, preds, target):
        # 计算原始方差特征图
        variance_map_preds0 = self.compute_variance_feature_map(preds, window_size_var=self.window_size0,
                                                                stride_var=self.stride0)
        variance_map_target0 = self.compute_variance_feature_map(target, window_size_var=self.window_size0,
                                                                 stride_var=self.stride0)
        l1_loss0 = self.loss(variance_map_preds0, variance_map_target0)
        # 计算四种新的特征图损失
        max4_mean_loss = self.loss(self.compute_max4_mean(preds), self.compute_max4_mean(target))
        min4_mean_loss = self.loss(self.compute_min4_mean(preds), self.compute_min4_mean(target))

        # 返回所有尺度的损失之和加上自定义特征图损失
        return l1_loss0*1  + max4_mean_loss*1 + min4_mean_loss*1

    def compute_variance_feature_map(self, image_tensor, window_size_var=8, stride_var=4):
        N, C, H, W = image_tensor.shape
        unfolded = f.unfold(image_tensor, kernel_size=window_size_var, stride=stride_var)
        unfolded = unfolded.view(N, C, window_size_var * window_size_var, -1)
        mean = unfolded.mean(dim=2, keepdim=True)
        variance = ((unfolded - mean) ** 2).mean(dim=2)
        output_height = (H - window_size_var) // stride_var + 1
        output_width = (W - window_size_var) // stride_var + 1
        variance_map = variance.view(N, C, output_height, output_width)
        return variance_map
    def compute_max4_mean(self, image_tensor):
        N, C, H, W = image_tensor.shape
        unfolded = f.unfold(image_tensor, kernel_size=4, stride=1)
        unfolded = unfolded.view(N, C, 16, -1)  # shape: [N, C, 16, num_windows]

        max4_mean = unfolded.topk(4, dim=2).values.mean(dim=2)  # 最大4个数的均值

        output_height = (H - 4) // 1 + 1
        output_width = (W - 4) // 1 + 1
        max4_mean_map = max4_mean.view(N, C, output_height, output_width)

        return max4_mean_map

    def compute_min4_mean(self, image_tensor):
        N, C, H, W = image_tensor.shape
        unfolded = f.unfold(image_tensor, kernel_size=4, stride=1)
        unfolded = unfolded.view(N, C, 16, -1)  # shape: [N, C, 16, num_windows]

        min4_mean = unfolded.topk(4, dim=2, largest=False).values.mean(dim=2)  # 最小4个数的均值

        output_height = (H - 4) // 1 + 1
        output_width = (W - 4) // 1 + 1
        min4_mean_map = min4_mean.view(N, C, output_height, output_width)

        return min4_mean_map


def train(model, train_dataloader, optimizer, criterion,device, save_dir, save_cycle, epochs, resume=None,
          ):
    if resume is not None:
        if not os.path.exists(resume):
            raise FileNotFoundError(f'Checkpoint {resume} not found.')
        else:
            print(f'Resume from {resume}')
            model.load_state_dict(torch.load(args.resume))
    else:
        print('Train from scratch...')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    best_loss = float("inf")

    content_loss = PerceptualLoss(torch.nn.MSELoss())
    content_loss.contentFunc.to(device)
    VarimpFeatureLoss = VarianceimpFeatureLoss().to(device)

    for epoch in range(epochs):
        losses = {'loss': [], 'loss1': [],'loss3': [],'loss7': []}
        for batch in tqdm(train_dataloader):
            LR_imgs = batch['LR_img'].to(device)
            HR_imgs = batch['HR_img'].to(device)

            SR_imgs = model(LR_imgs)

            loss1 = criterion(SR_imgs, HR_imgs)
            loss3 = content_loss.get_loss(SR_imgs, HR_imgs)
            loss7 = VarimpFeatureLoss.forward(SR_imgs, HR_imgs)
            loss = loss1 +loss3*0.001+0.1*loss7
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            losses['loss'].append(loss.item())
            losses['loss1'].append(loss1.item())
            losses['loss3'].append(loss3.item())
            losses['loss7'].append(loss7.item())
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        print('{} | Epoch:{}/{} | Loss:{:.5f} Loss1:{:.5f}Loss3:{:.5f}Loss7:{:.5f}'
              .format(current_time, epoch + 1, epochs, np.mean(losses['loss']), np.mean(losses['loss1']),np.mean(losses['loss3']),np.mean(losses['loss7'])
                      ))
        content = '{} | Epoch:{}/{} | Loss:{:.5f} Loss1:{:.5f}Loss3:{:.5f}Loss7:{:.5f}'.format(current_time, epoch + 1, epochs, np.mean(losses['loss']), np.mean(losses['loss1']),np.mean(losses['loss3']),np.mean(losses['loss7'])
                      )
        file_path_log = os.path.join(args.save_dir, 'output.txt')
        with open(file_path_log, 'a') as file:
            # 写入内容并换行
            file.write(content + '\n')

        if np.mean(losses['loss']) <= best_loss:
            torch.save(model.state_dict(), os.path.join(save_dir, 'best.pth'))
            best_loss = np.mean(losses['loss'])
            #print('Saving best model...')
        if (epoch + 1) % save_cycle == 0:
            torch.save(model.state_dict(), os.path.join(save_dir, '{}.pth'.format(epoch + 1)))
    print('\nTrain Complete.\n')

def main(args):
    dataset = SRDataset(root_dir=args.data_dir)
    train_dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=8)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    print(f'Using device: {device}')
    model=FAT().to(device)
    criterion = nn.L1Loss().to(device)

    optimizer = torch.optim.Adam(params=filter(lambda x: x.requires_grad, model.parameters()), lr=args.lr,
                                 betas=(0.9, 0.999),
                                 eps=1e-08)
    optimizer.zero_grad()
    subtitle = """  
         
          
          ___           ___           ___           ___           ___                                           
         /\  \         /\  \         /\  \         /\  \         /\  \                                          
        /::\  \       /::\  \        \:\  \       /::\  \       /::\  \                                         
       /:/\:\  \     /:/\:\  \        \:\  \     /:/\ \  \     /:/\:\  \                                        
      /::\~\:\  \   /::\~\:\  \       /::\  \   _\:\~\ \  \   /::\~\:\  \                                       
     /:/\:\ \:\__\ /:/\:\ \:\__\     /:/\:\__\ /\ \:\ \ \__\ /:/\:\ \:\__\                                      
     \/__\:\ \/__/ \/__\:\/:/  /    /:/  \/__/ \:\ \:\ \/__/ \/_|::\/:/  /                                      
          \:\__\        \::/  /    /:/  /       \:\ \:\__\      |:|::/  /                                       
           \/__/        /:/  /     \/__/         \:\/:/  /      |:|\/__/                                        
                       /:/  /                     \::/  /       |:|  |                                          
                       \/__/                       \/__/         \|__|                                          
          ___           ___           ___           ___           ___       ___           ___           ___     
         /\  \         /\  \         /\  \         /\  \         /\__\     /\__\         /\  \         /\__\    
        /::\  \       /::\  \       /::\  \       /::\  \       /:/  /    /:/  /        /::\  \       /:/  /    
       /:/\:\  \     /:/\:\  \     /:/\:\  \     /:/\:\  \     /:/  /    /:/  /        /:/\:\  \     /:/__/     
      /:/  \:\  \   /:/  \:\  \   /:/  \:\  \   /:/  \:\__\   /:/  /    /:/  /  ___   /:/  \:\  \   /::\__\____ 
     /:/__/_\:\__\ /:/__/ \:\__\ /:/__/ \:\__\ /:/__/ \:|__| /:/__/    /:/__/  /\__\ /:/__/ \:\__\ /:/\:::::\__\ 
     \:\  /\ \/__/ \:\  \ /:/  / \:\  \ /:/  / \:\  \ /:/  / \:\  \    \:\  \ /:/  / \:\  \  \/__/ \/_|:|~~|~   
      \:\ \:\__\    \:\  /:/  /   \:\  /:/  /   \:\  /:/  /   \:\  \    \:\  /:/  /   \:\  \          |:|  |    
       \:\/:/  /     \:\/:/  /     \:\/:/  /     \:\/:/  /     \:\  \    \:\/:/  /     \:\  \         |:|  |    
        \::/  /       \::/  /       \::/  /       \::/__/       \:\__\    \::/  /       \:\__\        |:|  |    
         \/__/         \/__/         \/__/         ~~            \/__/     \/__/         \/__/         \|__|      
    """

    # 使用print函数打印这个字符串
    print(subtitle)
    train(model, train_dataloader, optimizer, criterion,device, args.save_dir, args.save_cycle, args.epochs,
          args.resume)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a model.')#D:\yxylearn\IRfenbian\datasets\Sen480\cali D:\yxylearn\test\data\test\all  D:\yxylearn\test\data\test\BIT-SR24   D:\yxylearn\test\data\df2k
    parser.add_argument('--data_dir', type=str, default=r'D:\yxylearn\test\data\test\BIT-SR24', help='Path to dataset')
    parser.add_argument('--save_dir', type=str, default=r'./version/checkpoints6/realv1fewXU', help='Path to save checkpoints')
    parser.add_argument('--save_cycle', type=int, default=5, help='Cycle of saving checkpoint')
    parser.add_argument('--resume',default=r'./version/checkpoints6/realv1few/best.pth', type=str,help='Path to checkpoint file to resume training')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size of each data batch')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='Disable CUDA')
    parser.add_argument('--epochs', type=int, default=10, help='Epochs')
    args = parser.parse_args()
    main(args)
'''
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a model.')#D:\yxylearn\IRfenbian\datasets\Sen480\cali D:\yxylearn\test\data\test\all  D:\yxylearn\test\data\test\BIT-SR24
    parser.add_argument('--data_dir', type=str, default=r'D:\yxylearn\test\data\test\all', help='Path to dataset')
    parser.add_argument('--save_dir', type=str, default=r'./version/checkpoints3/allv5persr24loss0_05newv5', help='Path to save checkpoints')
    parser.add_argument('--save_cycle', type=int, default=20, help='Cycle of saving checkpoint')
    parser.add_argument('--resume', type=str,default=r'D:\yxylearn\test\fastformer\version\checkpoints3\sentarget2/best.pth',help='Path to checkpoint file to resume training')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size of each data batch')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='Disable CUDA')
    parser.add_argument('--epochs', type=int, default=30, help='Epochs')
    args = parser.parse_args()
    main(args)


'''
