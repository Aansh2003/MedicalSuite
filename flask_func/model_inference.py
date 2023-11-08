import torch
import cv2
from torchvision import *
import torch.nn as nn
from PIL import Image
from models.UNet import UNet
import numpy as np

def brainTumor(image):
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb_image)
    transformer = transforms.Compose([transforms.Resize((256,256)),transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    rgb_tensor = transformer(pil_image)

    model = models.resnet50(pretrained=False)
    nr_filters = model.fc.in_features
    model.fc = nn.Linear(nr_filters, 4)
    model.load_state_dict(torch.load('/home/aansh/wt-front/models/resnet50_4-way.pt',map_location=torch.device('cpu')))
    model.eval()
    rgb_tensor = rgb_tensor.unsqueeze(0)
    out = model(rgb_tensor)
    _,pred_t = torch.max(out, dim=1)
    prob = max((torch.softmax(out,dim=1)).tolist()[0])
    pred_t = int(pred_t)
    prob = float(prob)
    if pred_t == 2:
        return pred_t,prob,image
    
    model = UNet()
    model.load_state_dict(torch.load('/home/aansh/wt-front/models/unet_model2.pt',map_location=torch.device('cpu')))
    required_size = pil_image.size

    transformer = transforms.Compose([transforms.Resize((256,256)),transforms.ToTensor(),transforms.Normalize(mean=[0.5, 0.5, 0.5],std=[0.225, 0.225, 0.225])])
    final_transform = transforms.Compose([transforms.Resize((required_size[1],required_size[0])),transforms.Grayscale(),transforms.ToPILImage()])
    
    transformed_image = transformer(pil_image)
    model.eval()
    output = torch.round(model(transformed_image.unsqueeze(0)))
    for value in output:
        mask = final_transform(value)
    image = np.array(pil_image.convert('RGB'))
    mask = np.array(mask.convert('RGB'))
    color = np.array([0,255,0], dtype='uint8')
    masked_img = np.where(mask, color, image)
    out = cv2.addWeighted(image, 0.8, masked_img, 0.2,0)
    return pred_t,prob,out

def brain_tumor_parse(pred):
    if pred == 0:
        return 'Glioma'
    elif pred == 1:
        return 'Meningioma'
    elif pred == 2:
        return 'No Tumor'
    else:
        return 'Pituitary'
    
def retinal_scan(image):
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb_image)
    required_size = pil_image.size
    transformer = transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    rgb_tensor = transformer(pil_image)

    model = torch.hub.load('pytorch/vision:v0.10.0', 'densenet121', pretrained=True)
    model.classifier = nn.Sequential(
        nn.Linear(1024, 256),
        nn.Dropout(p=0.1),
        nn.ReLU(),
        nn.Linear(256, 4),
        nn.Softmax(dim=1)
    )
    model.load_state_dict(torch.load('/home/aansh/wt-front/models/dense_retinal.pt',map_location=torch.device('cpu')))

    rgb_tensor = rgb_tensor.unsqueeze(0)
    out = model(rgb_tensor)
    _,pred_t = torch.max(out, dim=1)
    prob = max((torch.softmax(out,dim=1)).tolist()[0])
    pred_t = int(pred_t)
    prob = float(prob)

    if pred_t == 3:
        return pred_t,prob,image
    
    model = UNet()
    model.load_state_dict(torch.load('/home/aansh/wt-front/models/vessel.pt',map_location=torch.device('cpu')))
    
    print(required_size)
    transform = transforms.Compose([
        transforms.Resize((256,256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                            std=[0.225, 0.225, 0.225])
    ])

    final_transform = transforms.Compose([
        transforms.Resize((required_size[1],required_size[0])),
        transforms.Grayscale(),
        transforms.ToPILImage()
    ])

    model.eval()
    transformed_image = transform(pil_image)
    output = torch.round(model(transformed_image.unsqueeze(0)))
    for value in output:
        mask = final_transform(value)
    pil_image = np.array(pil_image.convert('RGB'))
    mask = np.array(mask.convert('RGB'))

    color = np.array([200,0,0], dtype='uint8')

    image = cv2.cvtColor(pil_image,cv2.COLOR_RGB2BGR)
    mask = cv2.cvtColor(mask,cv2.COLOR_RGB2BGR)
    masked_img = np.where(mask, color, image)
    out = cv2.addWeighted(image, 0.8, masked_img, 0.2,0)
    return pred_t,prob,out

def retinal_parse(pred):
    if pred == 0:
        return 'Cataract'
    elif pred == 1:
        return 'Diabetic retinopathy'
    elif pred == 2:
        return 'Glaucoma'
    else:
        return 'No disease'

