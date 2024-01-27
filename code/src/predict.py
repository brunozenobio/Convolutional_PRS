from skimage import io
from PIL import Image
from torchvision import transforms
import torch
import sys
import torchvision


#### Defino la estructura del modelo #####
resnet = torchvision.models.resnet18()
class Model(torch.nn.Module):
  def __init__(self, n_outputs=5, pretrained=False, freeze=False):
    super().__init__()
    # descargamos resnet
    resnet = torchvision.models.resnet18(pretrained=pretrained)
    # nos quedamos con todas las capas menos la última
    self.resnet = torch.nn.Sequential(*list(resnet.children())[:-1])
    if freeze:
      for param in self.resnet.parameters():
        param.requires_grad=False
    # añado una nueva capa lineal para llevar a cabo la clasificación
    self.fc = torch.nn.Linear(512, 3)

  def forward(self, x):
    x = self.resnet(x)
    x = x.view(x.shape[0], -1)
    x = self.fc(x)
    return x

  def unfreeze(self):
    for param in self.resnet.parameters():
        param.requires_grad=True



def img_load(img_path):
    transform = transforms.Compose([

        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    img = io.imread(img_path)
    img = Image.fromarray(img[:, :, :3])
    img = transform(img)
    img = img.unsqueeze(0)   # Como el modelo recibe los batch debo hacer una dimension de batch.
    return img

    
def prediction(img_path):
    img_path = f'./data_test/{img_path}'
    img = img_load(img_path)
    
    #Cargo el modelo
    model = Model(pretrained=True,freeze=False)
    checkpoint = torch.load('./model/img_model.pth')
    model.load_state_dict(checkpoint['modelo'])
    model.eval()
    with torch.no_grad():
        predictions = model(img)
    predicted_classes = torch.argmax(predictions, dim=1)
    if predicted_classes == 0:
        return 'Paper'
    elif predicted_classes == 1:
        return 'Rock'
    else:
        return 'Scissors'



if __name__ == "__main__":
    # Obtener el parámetro de la línea de comandos
    parametro = sys.argv[1]
    prediction(parametro)
    
print(prediction(parametro))