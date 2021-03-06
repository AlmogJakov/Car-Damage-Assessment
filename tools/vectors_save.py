from PIL import Image
from torchvision import transforms, models
import torchvision
import torch
import pickle
import timeit

start = timeit.default_timer()
vgg2 = models.vgg16(pretrained=True)
device = torch.device("cuba" if torch.cuda.is_available() else "cpu")
vgg2.to(device)
n = 1512
vectors = []
for i in range(n):
    # load the images from folder one by one (Change to the desired input name)
    img_path = f'../database/image_224X224/{i}.jpeg'
    img = Image.open(img_path).convert('RGB')
    in_transform = transforms.Compose([
        torchvision.transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])])
    img_tensor = in_transform(img).unsqueeze(0)
    input_tensor = img_tensor.to(device)
    input_vgg = vgg2(input_tensor).detach().to('cpu')
    # print(input_vgg[0].tolist())
    vectors.append(input_vgg[0].tolist())

# Save the vectors to binary file (Change to the desired output name)
with open('../database/image_newDataBase_224X224.pkl', 'wb') as f:
    pickle.dump(vectors, f)

stop = timeit.default_timer()
print('Time: ', stop - start)
