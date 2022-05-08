import torch
from torchvision import models
import os
from glob import glob
import torchvision.transforms as transforms
from PIL import Image
from torch.autograd import Variable
#from img2vec_pytorch import Img2Vec
import pandas as pd

# Model
# Load the pretrained model
resnet_model = models.resnet18(pretrained=True)

model = torch.hub.load('ultralytics/yolov5', 'yolov5s') 
 # or yolov5m, yolov5l, yolov5x, custom

# Images

img_path =  './Images/'# or file, Path, PIL, OpenCV, numpy, list
#output_path = '/media/ashish-j/B/wheat_detection/flick_data/embeddings'
count=0
for file in os.listdir(img_path):
	img = os.path.join(img_path,file)
	results = model(img)
	results.crop(save_dir='./detection_data/'+str(img)+'/')
	
	

