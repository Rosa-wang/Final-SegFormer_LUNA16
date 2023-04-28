import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import argparse
import pandas as pd
import wandb
from torch import autocast
from torch.cuda.amp import GradScaler
from torch.autograd import Variable
from torchvision.ops import sigmoid_focal_loss
from model import Segformer
from configs import *
from dataloader import build_dataloader
from torchmetrics import Accuracy, Recall, Precision

def parse_args():
    parser = argparse.ArgumentParser("LUNA16 Segformer")

    # TRAINING
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--num_workers", default=0, type=int)
    parser.add_argument("--lr", default=5e-5, type=float)
    parser.add_argument("--epochs", default=100, type=int)
    parser.add_argument("--save_dir", default='./checkpoints', type=str)
    parser.add_argument("--save_freq", default=10, type=int)
    parser.add_argument("--grad_acc_step", default=8, type=int)
    return parser.parse_args()

class BCE_Dice_Loss(torch.nn.Module):
    def __init__(self, weight=1000, size_average=True):
        super(BCE_Dice_Loss, self).__init__()
        self.weight = weight
        #self.loss = nn.CrossEntropyLoss(reduction='mean')

    def forward(self, inputs, targets):
        one_hot_tgt = F.one_hot(targets, num_classes=2).permute(0,4,1,2,3).float()
        loss = sigmoid_focal_loss(inputs, one_hot_tgt).mean(dim=1)
        loss[targets!=0] *= self.weight
        loss = loss.mean()
        return loss


def dice_coeff(pred, target):
    smooth = 1.
    num = pred.size(0)
    m1 = pred.view(num, -1)  # Flatten
    m2 = target.view(num, -1)  # Flatten
    intersection = (m1 * m2).sum()
    return (2. * intersection + smooth) / (m1.sum() + m2.sum() + smooth)

def mIoU(pred, target):
    pred_ = pred.clone().argmax(1)
    target_ = target.clone()
    intersection = (pred_ * target_).sum(dim=(1,2,3))
    union = pred_.sum(dim=(1,2,3)) + target_.sum(dim=(1,2,3))
    if union.sum() == 0:
        return torch.tensor(torch.nan)
    iou = intersection / (union+1e-7)
    return iou.mean()


def train(args):
    model = Segformer(
        dims = (32, 64, 160, 256),      # dimensions of each stage
        heads = (1, 2, 5, 8),           # heads of each stage
        ff_expansion = (8, 8, 4, 4),    # feedforward expansion factor of each stage
        reduction_ratio = (8, 4, 2, 1), # reduction ratio of each stage for efficient attention
        num_layers = 2,                 # num layers of each stage
        channels = 1,                   # input channels
        decoder_dim = 256,              # decoder dimension
        num_classes = 2                 # number of segmentation classes
    )
    scaler = GradScaler()
    loss_fn = BCE_Dice_Loss()
    # loss_fn = nn.CrossEntropyLoss()
    grad_acc_step = args.grad_acc_step
    optimizer = optim.AdamW(model.parameters(), args.lr)
    train_loaders, val_loaders = build_dataloader()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)  # 将模型移到GPU上
    fold = 0
    
    accuracy_metric = Accuracy(task="binary").to(device)
    recall_metric = Recall(task="binary", average='macro').to(device)
    prec_metric = Precision(task="binary", average='macro').to(device)

    for train_loader, val_loader in zip(train_loaders, val_loaders):
        data_len = len(train_loader.dataset)
        for epoch in range(args.epochs):
            model.train()  # set the model to training mode
            interval = 20
            valid_interval = interval
            train_loss = 0.0
            iter_loss = .0
            acc, prec, recall = [0.]*3
            train_dice, iter_dice = 0.0, 0.0
            train_miou, iter_miou = 0.0, 0.0
            for batch_idx, (data, target) in enumerate(train_loader):
                optimizer.zero_grad()
                if torch.cuda.is_available():
                    data = Variable(data.cuda())
                    target = Variable(target.cuda())
                data = data.to(torch.float16).to(device)  # 将输入数据转换为float32类型，并移到GPU上
                target = target.to(torch.long).to(device)  # 将输入数据转换为float32类型，并移到GPU上   

                with autocast(device_type='cuda', dtype=torch.float16):
                    logits = model(data) # Model forward
                    loss = loss_fn(logits, target) # Loss function 
                    loss = loss / grad_acc_step

                # if loss.isnan().sum() > 0:
                #     import pdb; pdb.set_trace()
                torch.nn.utils.clip_grad_value_(model.parameters(), 5)
                scaler.scale(loss).backward()

                if ((batch_idx+1) % grad_acc_step == 0):
                    scaler.step(optimizer)
                    scaler.update()

                train_loss += loss.item() 
                iter_loss += loss.item()
                miou = mIoU(logits, target)
                if not miou.isnan().sum():
                    train_miou += miou
                    iter_miou += miou
                else: 
                    valid_interval -= 1
                
                pred = logits.argmax(dim=1)
                acc_ = accuracy_metric(pred, target)
                prec_ = prec_metric(pred, target)
                recall_ = recall_metric(pred, target)

                acc += acc_
                prec += prec_
                recall += recall_
                if (batch_idx+1) % interval == 0:
                    step = batch_idx+1
                    progress = 100. * batch_idx / data_len
                    # iter_dice /= step
                    iter_loss /= interval
                    iter_miou = iter_miou / valid_interval * 100
                    acc = acc / interval * 100
                    prec = prec / interval * 100
                    recall = recall / interval * 100

                    wandb.log({'accuracy': acc, 
                                'precision': prec,
                                'miou': iter_miou,
                                'recall': recall,
                                'loss': iter_loss})

                    log_info = f'Train Epoch: {epoch+1} [{step}/{data_len} ({progress:.2f}%)]\t'
                    log_info += f'Loss: {(iter_loss):.4f}\t'
                    log_info += f'mIoU: {iter_miou:.4f}%\t'
                    log_info += f'Acc: {acc:.4f}%\t'
                    log_info += f'Precision: {prec:.4f}%\t'
                    log_info += f'Recall: {recall:.4f}%\t'
                    print(log_info)

                    valid_interval = interval
                    iter_loss = .0
                    iter_miou = .0
                    acc = .0
                    prec = .0
                    recall = .0

            train_loss /= data_len
            train_miou /= data_len
            # train_acc = 100 * correct / total
            print(f'Train Epoch: {epoch+1} \t No: {fold} \t Loss: {train_loss:.6f} \t Miou: {100. * train_miou:.3f}%' )
            model.eval()
            # with torch.no_grad():
            #     val_loss = 0
            #     for batch, (t_data, t_target) in enumerate(val_loader):
            #         t_data = t_data.to(torch.float32).to(torch.device("cuda"))  # 将输入数据转换为float32类型，并移到GPU上
            #         t_target = t_target.to(torch.float32).to(torch.device("cuda"))  # 将输入数据转换为float32类型，并移到GPU上    
            #         outputs = model(t_data) # Model forward
            #         loss = loss_fn(F.softmax(outputs, dim=1)[:, 1], t_target)
            #         val_loss += loss.item() * len(t_target)
            #     val_loss /= len(val_loader.dataset)
            #     print(f'Validation Loss: {val_loss:.4f}')
            if (epoch + 1) % args.save_freq == 0:
                torch.save(model.state_dict(), f"{args.save_dir}/fold{fold}model_epoch_{epoch + 1}.pt")
        fold += 1    
             

    
# def test(args):
#     test_dataset = LUNA16_dataset(args)
#     test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
#     size = len(test_dataset)
#     num_batches = len(test_loader)
#     test_loss, correct = 0, 0

#     with torch.no_grad():
#          for epochs in range(args.epochs):
#             for batch_idx, batch in enumerate(test_loader):
#                 import pdb; pdb.set_trace()
#                 pred = model(batch)
#                 test_loss += loss_fn(pred,batch['label'] ).item()
#                 correct += (pred.argmax(1) == y).type(torch.float).sum().item()

#     test_loss /= num_batches
#     correct /= size
#     print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    

if __name__ == "__main__":
    args = parse_args()
    wandb.init(project='LUNA-16')
    train(args)
