import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from segformer_pytorch import Segformer
import SimpleITK as sitk
import matplotlib.pyplot as plt
import pandas as pd
from glob import glob
from torchvision import transforms
import csv
import os
from tqdm import tqdm_notebook as tq

def load_itk_image(filename):
    itkimage = sitk.ReadImage(filename)
    numpyImage = sitk.GetArrayFromImage(itkimage)
   
    numpyOrigin = np.array(list(reversed(itkimage.GetOrigin())))
    numpySpacing = np.array(list(reversed(itkimage.GetSpacing())))
    return numpyImage, numpyOrigin, numpySpacing

def get_filename(case):
    global file_list
    for f in file_list:
        if case in f:
            return(f)      
        
def make_mask(center,diam,z,width,height,spacing,origin):
    mask = np.zeros([height,width]) # 0's everywhere except nodule swapping x,y to match img
    #convert to nodule space from world coordinates

    # Defining the voxel range in which the nodule falls
    v_center = (center-origin)/spacing
    v_diam = int((diam+5)/spacing[0])
    v_xmin = np.max([0,int(v_center[0]-v_diam)])
    v_xmax = np.min([width-1,int(v_center[0]+v_diam)])
    v_ymin = np.max([0,int(v_center[1]-v_diam)]) 
    v_ymax = np.min([height-1,int(v_center[1]+v_diam)])

    v_xrange = range(v_xmin,v_xmax+1)
    v_yrange = range(v_ymin,v_ymax+1)

    # Convert back to world coordinates for distance calculation
    x_data = [x*spacing[0]+origin[0] for x in range(width)]
    y_data = [x*spacing[1]+origin[1] for x in range(height)]
    for v_x in v_xrange:
        for v_y in v_yrange:
            p_x = spacing[0]*v_x + origin[0]
            p_y = spacing[1]*v_y + origin[1]
            if np.linalg.norm(center-np.array([p_x,p_y,z]))<=diam:
                mask[int((p_y-origin[1])/spacing[1]),int((p_x-origin[0])/spacing[0])] = 1.0
    return(mask)

def matrix2int16(matrix):
    ''' 
matrix must be a numpy array NXN
Returns uint16 version
    '''
    m_min= np.min(matrix)
    m_max= np.max(matrix)
    matrix = matrix-m_min
    return(np.array(np.rint( (matrix-m_min)/float(m_max-m_min) * 65535.0),dtype=np.uint16))

luna_path="/LUNA16/"
luna_subset_path = '/LUNA16/test/'
result_path = '/Results/img_result/'
seg_model_loadPath = '/Results/segformer result/result0/'
netS =  Segformer(
	dims = (32, 64, 160, 256),      # dimensions of each stage
	heads = (1, 2, 5, 8),           # heads of each stage
	ff_expansion = (8, 8, 4, 4),    # feedforward expansion factor of each stage
	reduction_ratio = (8, 4, 2, 1), # reduction ratio of each stage for efficient attention
	num_layers = 2,                 # num layers of each stage
	channels = 1,                   # input channels
	decoder_dim = 256,              # decoder dimension
	num_classes = 2                 # number of segmentation classes
)

netS.load_state_dict(torch.load(seg_model_loadPath+'seg_best.pt'))

apply_norm = transforms.Normalize([-460.466],[444.421]) 

fcount = 0
file_list=glob(luna_subset_path+"*.mhd")
df_node = pd.read_csv(luna_path+"annotations.csv")
df_node["file"] = df_node["seriesuid"].apply(get_filename)
df_node = df_node.dropna()
for img_file in tq(file_list):       
    mini_df = df_node[df_node["file"]==img_file] 
    if len(mini_df)>0:       # some files may not have a nodule--skipping those 
        biggest_node = np.argsort(mini_df["diameter_mm"].values)[-1]   # just using the biggest node
        node_x = mini_df["coordX"].values[biggest_node]
        node_y = mini_df["coordY"].values[biggest_node]
        node_z = mini_df["coordZ"].values[biggest_node]
        diam = mini_df["diameter_mm"].values[biggest_node]

        itk_img = sitk.ReadImage(img_file) 
        img_array = sitk.GetArrayFromImage(itk_img) # indexes are z,y,x (notice the ordering)
        center = np.array([node_x,node_y,node_z])   # nodule center
        origin = np.array(itk_img.GetOrigin())      # x,y,z  Origin in world coordinates (mm)
        spacing = np.array(itk_img.GetSpacing())    # spacing of voxels in world coor. (mm)
        v_center =np.rint((center-origin)/spacing)  # nodule center in voxel space (still x,y,z ordering)
        num_z, height, width = img_array.shape

        center = np.array([node_x, node_y, node_z])   # nodule center
        v_center = np.rint((center-origin)/spacing) 

        for i_z in range(int(v_center[2])-1,int(v_center[2])+2):
            mask = make_mask(center,diam,i_z*spacing[2]+origin[2],width,height,spacing,origin)
            masks = mask
            imgs = img_array[i_z]
            middle = imgs[100:400,100:400]
            mean = np.mean(middle)  
            max = np.max(imgs)
            min = np.min(imgs)
            imgs[imgs==max]=mean
            imgs[imgs==min]=mean
            # np.save(result_path+"images_"+str(fcount)+str(mini_df["seriesuid"].unique())+".npy",imgs)
            # np.save(result_path+"masks_"+str(fcount)+str(mini_df["seriesuid"].unique())+".npy",masks)
            img_s = torch.from_numpy(imgs).unsqueeze(0).float()
            mid_mean = img_s[:,100:400,100:400].mean()    
            img_s[img_s==img_s.min()] = mid_mean
            img_s[img_s==img_s.max()] = mid_mean
            imgs_norm = apply_norm(img_s).unsqueeze(0)
            out = F.softmax(netS(imgs_norm),dim=1)
            out_np = np.asarray(out[0,1].squeeze(0).detach().cpu().numpy()*255,dtype=np.uint8)
    
            ret, thresh = cv2.threshold(out_np,0,1,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
            connectivity = 4  
            output = cv2.connectedComponentsWithStats(thresh, connectivity, cv2.CV_32S)
            stats = output[2]
            temp = stats[1:, cv2.CC_STAT_AREA]
            if len(temp)>0:
                largest_label = 1 + np.argmax(temp)    
                areas = stats[1:, cv2.CC_STAT_AREA]
                max_area = np.max(areas)
                if max_area>150:
                    plt.figure(figsize=[15,5])
                    plt.subplot(131)
                    plt.imshow(img_s.squeeze(0).squeeze(0).numpy(),cmap='gray')
                    plt.title('Original image')
                    plt.subplot(132)
                    plt.imshow(out[0,1].squeeze(0).detach().numpy(),cmap='gray')
                    plt.title('Segmented regions')
                    plt.subplot(133)
                    plt.imshow((imgs)*masks,cmap='gray')
                    plt.title('lable mask')
                    plt.savefig(result_path+"res_"+str(fcount)+str(mini_df["seriesuid"].unique())+'.png')
                    plt.close()

            fcount +=1


