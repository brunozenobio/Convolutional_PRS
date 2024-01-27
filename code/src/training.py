
#Librerias utiles para los archivos y las imagenes
import os
import numpy as np
from skimage import io
from PIL import Image
from torchvision import transforms

#Librerias para el modelo
import torchvision
import torch
from tqdm import tqdm
from sklearn.model_selection import train_test_split

#Funcion que realiza el entrenamiento.
from fit import fit


############################# CARGO LAS RUTAS DE LAS IMAGENES ############################

images_path = './data_train/' # Ruta de las imagenes
classes = os.listdir(images_path) # Carpetas con cada label.
print(f'Clases : {classes}')

##########################################################################################

################ GENERO LAS LISTAS CON LAS RUTAS DE IAMGENES Y LOS LABELS ################

images,label = [],[] # Genero 2 listas con la ruta relativa de la imagen incluida esta, y otra con los labeles,
for ind,clase in enumerate(classes):
    direccion = os.listdir(f'{images_path}{clase}')
    images += [f'{images_path}{clase}/{img}' for img in direccion]
    label += [ind]*len(direccion)

############################################################################################


#################### PREPARO LOS DATOS PARA INGRESAR AL MODELO ############################

device = "cuda" if torch.cuda.is_available() else "cpu"
train_images, test_images, train_labels, test_labels = train_test_split(images, label, test_size=0.2, random_state=42)


class Dataset(torch.utils.data.Dataset): # Genero la clase Dataset para las imagenes
  def __init__(self, X, y, trans, device):
    self.X = X # Conjunto de imagenes
    self.y = y # Clases
    self.trans = trans # Trasnformacion que pueda aplicarle
    self.device = device

  def __len__(self):
    return len(self.X)

  def __getitem__(self, ix):
    # cargar la imágen
    img = io.imread(self.X[ix]) #Leo la imagen
    img = Image.fromarray(img)
    # aplicar transformaciones
    if self.trans: # Si incorpore una trasnformacion la aplico
      img = self.trans(img)
    return img, torch.tensor(self.y[ix]) #Retorno cfomo tensor, con la imagen normalizada y permutadando dfebido a los canales

transform = transforms.Compose([

    transforms.Resize((224, 224)),
    transforms.ToTensor(),

]) # Realizo una trasnformacion para escalar las imagenes a 224x224


# Diccionario dataset, que contiene un objeto dataset para train y test
dataset = {
    'train': Dataset(train_images, train_labels, transform, device),
    'test': Dataset(test_images, test_labels, transform, device),

}

dataloader = {
    'train': torch.utils.data.DataLoader(dataset['train'], batch_size=64, shuffle=True, pin_memory=True),
    'test': torch.utils.data.DataLoader(dataset['test'], batch_size=256, shuffle=False)
}


################################################################################


######################## REALIZO EL ENTRENAMIENTO ###############################

#Importo el modelo resnet, el cual es un modelo preentrenado con 1000 salidas
resnet = torchvision.models.resnet18()

class Model(torch.nn.Module):
  def __init__(self, n_outputs=3, pretrained=False, freeze=False):
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


model = Model(pretrained=True,freeze=False)
epochs = 4
fit(model, dataloader, epochs)

#Finalmente guardo el modelo lo guardo en formato para poder reentrenar si quisiera

checkpoint = {
    'modelo': model.state_dict(),
    'optimizador': torch.optim.SGD(model.parameters(), lr=1e-2).state_dict(),
    'epocas': epochs
}

torch.save(checkpoint,'./model/img_model.pth' )