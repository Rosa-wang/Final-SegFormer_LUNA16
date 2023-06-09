#Code written by Rakshith Sathish
#The work is made public with MIT License


import numpy as np
import torch
import torch.nn as nn
from torch import optim
import tqdm
import time
from torch.utils import data
import os
import wandb
import torch.nn
import torch.nn.functional as F
from torch.autograd import  Variable
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from sklearn.metrics import confusion_matrix
from segformer_pytorch import Segformer
from LUNA_loader import lunaLoader
import lovasz_losses as L

def dice_coefficient(pred, target):
	predC = torch.argmax(F.softmax(pred,dim=1),dim=1)
	c = confusion_matrix(target.view(-1).cpu().numpy(), predC.view(-1).cpu().numpy(),labels=[0,1])
	TP = np.diag(c)
	FP = c.sum(axis=0) - np.diag(c)  
	FN = c.sum(axis=1) - np.diag(c)
	TN = c.sum() - (FP + FN + TP)
	return (TP,FP,FN)



savePath = 'Results/segformer result/result0/'
if not os.path.isdir(savePath):
	os.makedirs(savePath)


trainDset = lunaLoader(is_transform=True, split='train',img_size=512)
valDset = lunaLoader(is_transform=True, split='val',img_size=512)

trainDataLoader = data.DataLoader(trainDset,batch_size=8,shuffle=True,num_workers=4,pin_memory=True)
validDataLoader = data.DataLoader(valDset,batch_size=8,shuffle=False,num_workers=4,pin_memory=True)

n_classes = 2
net =  Segformer(
	dims = (32, 64, 160, 256),
	# dims = (64, 128, 320, 512),# dimensions of each stage
	heads = (1, 2, 5, 8),           # heads of each stage
	ff_expansion = (8, 8, 4, 4),    # feedforward expansion factor of each stage
	reduction_ratio = (8, 4, 2, 1), # reduction ratio of each stage for efficient attention
	num_layers = 2,                 # num layers of each stage
	channels = 1,                   # input channels
	decoder_dim = 256,              # decoder dimension
	num_classes = 2                 # number of segmentation classes
)
use_gpu = torch.cuda.is_available()
if use_gpu:
	net = net.cuda()

wandb.init(project="Segformer", name="2d-Segformer")
# optimizerS = optim.Adam(net.parameters(), lr = 2e-4, weight_decay = 1e-5)
optimizerS = optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-5)
scheduler = optim.lr_scheduler.StepLR(optimizerS, step_size=15, gamma=0.3)
criterionS = nn.CrossEntropyLoss()

epochs = 100
trainLoss = []
validLoss = []
trainDiceCoeff = []
validDiceCoeff = []
trainIoU = []
validIoU = []
start = time.time()

bestValidDice = 0.0

for epoch in range(epochs):
	epochStart = time.time()
	trainRunningLoss = 0
	validRunningLoss = 0
	trainBatches = 0
	validBatches = 0

	train_tp = np.zeros(n_classes)
	train_fp = np.zeros(n_classes)
	train_fn = np.zeros(n_classes)

	val_tp = np.zeros(n_classes)
	val_fp = np.zeros(n_classes)
	val_fn = np.zeros(n_classes)
	 

	net.train(True)
	for data1 in tqdm.tqdm(trainDataLoader):
		imgs, mask = data1 
		# print(imgs.shape)
		if use_gpu:
			inputs = imgs.cuda()
			labels = mask.cuda()

		
		cpmap = net(Variable(inputs))
		cpmapD = F.softmax(cpmap,dim=1)        

		LGce = criterionS(cpmap,labels.long()) 
		L_lov = L.lovasz_softmax(F.softmax(cpmap,dim=1),labels)
		LGseg = LGce*10+L_lov
		wandb.log({"train_loss": LGseg.item()})

		optimizerS.zero_grad()
		LGseg.backward() 
		optimizerS.step()                    

		trainRunningLoss += LGseg.item()

		train_cf = dice_coefficient(cpmapD,labels)        
		train_tp += train_cf[0]
		train_fp += train_cf[1]
		train_fn += train_cf[2]          
		trainBatches += 1  
		# break
		
	train_dice = (2*train_tp)/(2*train_tp + train_fp + train_fn )
	train_iou = (train_tp)/(train_tp + train_fp + train_fn )
	trainLoss.append(trainRunningLoss/trainBatches)
	trainDiceCoeff.append(train_dice)
	trainIoU.append(train_iou)

	print("\n{}][{}]| LGseg: {:.4f} | "
		.format(epoch,epochs,LGseg.item()))

	with torch.no_grad():
		for data1 in tqdm.tqdm(validDataLoader):
			imgs, mask = data1 
			if use_gpu:
				inputs = imgs.cuda()
				labels = mask.cuda()	

			
			cpmap = net(Variable(inputs))
			cpmapD = F.softmax(cpmap.data,dim=1)  

			val_cf = dice_coefficient(cpmapD,labels)        
			val_tp += val_cf[0]
			val_fp += val_cf[1]
			val_fn += val_cf[2]                      
			validRunningLoss += LGseg.item()
			validBatches += 1 
			# break   
			

		val_dice = (2*val_tp)/(2*val_tp + val_fp + val_fn)
		val_iou = (val_tp)/(val_tp + val_fp + val_fn)
		validLoss.append(validRunningLoss/validBatches)
		validDiceCoeff.append(val_dice)
		validIoU.append(val_iou)
	# scheduler.step(validRunningLoss/validBatches)
	if (val_dice[1] > bestValidDice):
		bestValidDice = val_dice[1]
		torch.save(net.state_dict(), savePath+'seg_best.pt')
	scheduler.step()	
   
	plt.figure()
	plt.plot(range(len(trainLoss)),trainLoss,'-r',label='Train')
	plt.plot(range(len(validLoss)),validLoss,'-g',label='Valid')
	if epoch==0:
		plt.legend()
	plt.savefig(savePath+'LossPlot.png')
	plt.close()
	epochEnd = time.time()-epochStart
	print('Epoch: {:.0f}/{:.0f} | Train Loss: {:.3f} | Valid Loss: {:.3f}'\
		  .format(epoch+1, epochs, trainRunningLoss/trainBatches, validRunningLoss/validBatches))
	print('\nDice | Train  | Dice {:.3f} | mIoU {:.3f} |\n Valid | Dice {:.3f} | mIoU {:.3f} |'
		  .format(np.mean(train_dice), np.mean(train_iou), np.mean(val_dice), np.mean(val_iou)))

	print('\nTime: {:.0f}m {:.0f}s'.format(epochEnd//60,epochEnd%60))
	trainLoss_np = np.array(trainLoss)
	validLoss_np = np.array(validLoss)
	trainDiceCoeff_np = np.array(trainDiceCoeff)
	validDiceCoeff_np = np.array(validDiceCoeff)
	trainIoU_np = np.array(trainIoU)
	validIoU_np = np.array(validIoU)

	wandb.log({"train_loss": trainRunningLoss/trainBatches, "valid_loss": validRunningLoss/validBatches})
	wandb.log({"train_Dice": np.mean(train_dice), "valid_Dice": np.mean(val_dice), "train_mIoU": np.mean(train_iou), "val_mIoU": np.mean(val_iou)})
	print('Saving losses')

	torch.save(trainLoss_np, savePath+'trainLoss.pt')
	torch.save(validLoss_np, savePath+'validLoss.pt')
	torch.save(trainDiceCoeff_np, savePath+'trainDice.pt')
	torch.save(validDiceCoeff_np, savePath+'validDice.pt')
	torch.save(trainIoU_np, savePath+'trainIoU.pt')
	torch.save(validIoU_np, savePath+'validIoU.pt')
	# break

	
end = time.time()-start
print('Training completed in {:.0f}m {:.0f}s'.format(end//60,end%60))
plt.figure()
plt.plot(range(len(trainLoss)),trainLoss,'-r')
plt.plot(range(len(validLoss)),validLoss,'-g')
plt.title('Loss plot')
plt.savefig(savePath+'trainLossFinal.png')
plt.close()

# trainDiceCoeff_bg = [x[0] for x in trainDiceCoeff]
# trainDiceCoeff_nodule = [x[1] for x in trainDiceCoeff]
# plt.figure()
# plt.plot(range(len(trainDiceCoeff_bg)),trainDiceCoeff_bg,'-r',label='BG')
# plt.plot(range(len(trainDiceCoeff_nodule)),trainDiceCoeff_nodule,'-g',label='Nodule')
# plt.legend()
# plt.title('Dice coefficient: Train')
# plt.savefig(savePath+'trainDice.png')
# plt.close()

# validDiceCoeff_bg = [x[0] for x in validDiceCoeff]
# validDiceCoeff_nodule = [x[1] for x in validDiceCoeff]
# plt.figure()
# plt.plot(range(len(validDiceCoeff_bg)),validDiceCoeff_bg,'-r',label='BG')
# plt.plot(range(len(validDiceCoeff_nodule)),validDiceCoeff_nodule,'-g',label='Nodule')
# plt.legend()
# plt.title('Dice coefficient: Valid')
# plt.savefig(savePath+'validDice.png')
# plt.close()
