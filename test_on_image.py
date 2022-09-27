# Test the model on a random image from the internet
from importlib.resources import path
import os
from matplotlib.image import pil_to_array 
import numpy as np 
import rasterio as rio
from torchvision import models
from torchvision.transforms import transforms  
import torch.nn as nn
from model import TextGenerator_LSTM, TextGenerator_GRU
import torch
from utils import *
from main import generator
import cv2


PATH = "cityimg3.webp"

senteces_path = r'questions\UCM_dataset\filenames\descriptions_UCM.txt'
train_filenames_path = r'questions\UCM_dataset\filenames\filenames_train.txt'
val_filenames_path = r'questions\UCM_dataset\filenames\filenames_val.txt'
test_filenames_path = r'questions\UCM_dataset\filenames\filenames_test.txt'

sentences, max_len = convert_to_words_ucm(senteces_path)
# Sentences is a dictionary with key as the image name and value as the sentence

# Split into train_test_val 
train_,val_,test_ = create_lists_ucm_uav(train_filenames_path,val_filenames_path,test_filenames_path)

train_sentences = [sentences[i] for i in train_]
test_sentences = [sentences[i] for i in test_]
val_sentences = [sentences[i] for i in val_]

# Create the dictionary with train and val sentences
value_to_idx,idx_to_value,ignored_words = word_frequency_ucm_uav(list(chain(*train_sentences, *val_sentences)), 5, test_sentences)

feature_extractor = models.resnet152(pretrained=True)
modules = list(feature_extractor.children())[:-1]      # RESNET 152
feature_extractor = nn.Sequential(*modules)         # RESNET 152
feature_extractor.eval()
feature_extractor.cuda()

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

pil_img = cv2.imread(os.path.join(PATH),cv2.IMREAD_COLOR)


pil_img = torch.from_numpy(pil_img).type(torch.FloatTensor).cuda().permute(2,0,1).unsqueeze(0)
pil_img = pil_img/255
pil_img = normalize(pil_img)

features = feature_extractor(pil_img).squeeze()

model = TextGenerator_GRU(len(value_to_idx.keys()),256,2048)
    
print('Loading model..')
model.load_state_dict(torch.load(r'weights\word\ucm\textGenerator_GRU_resnet152.pt'))
if(torch.cuda.is_available()):
    model.cuda()

prediction = generator(model,features,idx_to_value,value_to_idx,30)

print(prediction)