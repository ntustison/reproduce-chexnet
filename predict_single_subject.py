import torch
from PIL import Image
from torchvision import transforms, models
import pandas as pd

# model_file = "/home/ntustison/Data/reproduce-chexnet/results/checkpoint"
weights_filename = "/home/ntustison/Data/reproduce-chexnet/chexnet_repro_pytorch.h5"
image_file = "/home/ntustison/Data/reproduce-chexnet/data/images/00000001_001.png"

disease_categories = ['Atelectasis',
                      'Cardiomegaly',
                      'Effusion',
                      'Infiltration',
                      'Mass',
                      'Nodule',
                      'Pneumonia',
                      'Pneumothorax',
                      'Consolidation',
                      'Edema',
                      'Emphysema',
                      'Fibrosis',
                      'Pleural_Thickening',
                      'Hernia']

# checkpoint_best = torch.load(model_file, map_location=torch.device('cpu'))
# model = checkpoint_best['model']
model = models.densenet121(weights='DEFAULT')
model.classifier = torch.nn.Sequential(torch.nn.Linear(model.classifier.in_features, 
                                                       len(disease_categories)), 
                                       torch.nn.Sigmoid())
model.eval()
model.load_state_dict(torch.load(weights_filename))

image = Image.open(image_file)
image = image.convert('RGB')

# use imagenet mean,std for normalization
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
data_transforms = transforms.Compose([transforms.Resize(224),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize(mean, std)
                                    ])
batchX = data_transforms(image)
batchX = batchX[None, :]
batchY = model(batchX)

diagnosis_df = pd.DataFrame(batchY.detach().numpy(), columns=disease_categories)
print(diagnosis_df)


