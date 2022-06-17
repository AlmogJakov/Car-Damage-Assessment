# import cv2
from PIL import Image
# import matplotlib.pyplot as plt
from torchvision import transforms, models
import torchvision
import torch

import timeit
start = timeit.default_timer()

vgg2 = models.vgg16(pretrained=True)
device = torch.device("cuba" if torch.cuda.is_available() else "cpu")
vgg2.to(device)
print('itay')
n = 1512
for i in range(5):
    # load an image from file
    img_path = f'database/image_224X224/{i}.jpg'
    # img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB) / 255
    img = Image.open(img_path).convert('RGB')
    in_transform = transforms.Compose([
        torchvision.transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])])
    img_tensor = in_transform(img).unsqueeze(0)
    # # print(img_tensor.shape)
    #
    input_tensor = img_tensor.to(device)
    input_vgg = vgg2(input_tensor).detach().to('cpu')
    # # print(input_vgg)

stop = timeit.default_timer()
print('Time: ', stop - start)