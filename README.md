<p align="center"><img src="https://user-images.githubusercontent.com/67664604/217914153-1eb00e25-ac08-4dfa-aaf8-53c09038f082.png"></p>

<h1 align='center'> Proyectos: Detección de imagenes de piedra papel o tijeras con redes neuronales convolucionales</h1>


<h2 align='center'>Bruno Zenobio, DATAFT16</h2>

---

## **`Tabla de Contenidos`**

- [Introducción](#introducción)
- [Desarrollo](#desarrollo)
    - [Extracción de los datos](#extracción-de-los-datos)
    - [Preparación](#preparación-de-los-datos)
    - [Modelos](#modelosn)
    - [Predicciones](#despliegue-para-la-api)
- [Contacto](#contacto)

- ## **`Links`**
    - [Carpeta con los dataset](./datasets/)
    - [Proceso de ETL y EDA](./ETL%20y%20EDA/)
    - [Modelo de Recomendación](./model/)
    - [API desplegada en Render](https://steam-api-deploy.onrender.com/docs)
    - [Link al video](https://www.youtube.com/watch?v=IrYgvdtJEaI)



---

# Introducción

En este proyecto, me sumergire en el mundo de las redes neuronales, para realizar un modelo de predicción de imagenes sobre el juego piedra papel o tijera, para esto evaluaremos distintos modelos alterados de la arquitectura ResNet:

Las redes ResNet usan una arquitectura de red neuronal residual, de esta forma se aprende a mapear la entrada a la diferencia entre la entrada y la salida deseada. Utilizando capas convolucionales y de Pooling en esa arquitectura se puede obtener un resultado bastante aceptable sin contar con una gran cantidad de datos, a partir de la tecnica de transfer learning.

<p align="center"><img src="./images/The-ResNet.jpg"></p>

Ahora bien, entrando de lleno en el modelo, este tiene por objeto probar diferentes estrategias, dentro de la arquitectura ResNet, a partir de un dataset que no posee una gran cantidad de imagenes, este mismo fue extraido de Kaggle [dataset](https://www.kaggle.com/datasets/glushko/rock-paper-scissors-dataset) y guarda imagenes, de manos realizando los gestos de piedra,papel y tijeras.

## INSERTAR IMAGENES DE MUESTRA

<p align="center"><img src="./images/Diccionario.jpg"></p>

---

# Desarrollo

### Extracción de los datos

Como antes se habia mensionado, se extrayeron de la plataforma de Kaggle, obteniendose un total de (insertar cantidad), los cuales fueron disponibilizados para el entrenamiento y testeo.


### Preparación de los datos

Como se usara la la libreria de PyTorch, esta necesita a los datos en cierto formato, ademas hay que tener cuidado con el orden de la dimension de una imagen, ya que suelen venir en (alto x ancho x canal) y se necesita (canal x alto x ancho), es por esto que se define una clase Dataset, la cual extiende de PyTorch y nos permitira generar el conjunto de datos adecuados, en este se aplican transformaciónes necesarias en la imagen para que quede formato adecuado. 

Ademas con el proposito de realizar un entrenamiento mas optimo se debe separar las imagenes en batches, esto se hara con dataloader. Finalmente nuestro conjunto de datos quedara de la forma (batch_size x canal x ancho x alto).


### Modelos

Para poder probar, distintas estructuras, a partir de la arquitectura de ResNet, modificada para que las clases de salida sean un total de 3, se probaran distintas estrategias con una cantidad de 8 iteraciones.

- **`Arquitecura de ResNet sin sus pesos`**:
- **`Arquitecura de ResNet con sus pesos fijos(pretrainet = True,freeze=True)`**:
- **`Arquitecura de ResNet con sus pesos alterables(pretrainet = True,freeze=False`**:

### Despliegue para la API

Se desarrollaron las siguientes funciones, a las cuales se podrá acceder desde la API en la página Render:

- **`developer(desarrollador: str)`**: Retorna la cantidad de ítems y el porcentaje de contenido gratis por año para un desarrollador dado.
- **`userdata(User_id: str)`**: Retorna el dinero gastado, cantidad de ítems y el porcentaje de comentarios positivos en la revisión para un usuario dado.
- **`UserForGenre(género: str)`**: Retorna al usuario que acumula más horas para un género dado y la cantidad de horas por año.
- **`best_developer_year(año: int)`**: Retorna los tres desarrolladores con más juegos recomendados por usuarios para un año dado.
- **`developer_rec(desarrolladora: str)`**: Retorna una lista con la cantidad de usuarios con análisis de sentimiento positivo y negativo para un desarrollador dado.
- **`ser_recommend(user:str)`**: Esta función recomienda 5 juegos para un usuario especificado usando un filtro colaborativo.
- **`item_recommend(item:int)`**: Esta función recomienda 5 ítems dado un ítem específico usando un filtro basado en contenido.


## Contacto

<div style="display: flex; align-items: center;">
  <a href="https://www.linkedin.com/in/brunozenobio/" style="margin-right: 10px;">
    <img src="./images/in_linked_linkedin_media_social_icon_124259.png" alt="LinkedIn" width="40" height="40">
  </a>
  <a href="mailto:brunozenobio4@gmail.com" style="margin-right: 10px;">
    <img src="./images/gmail_new_logo_icon_159149.png" alt="Gmail" width="40" height="40">
  </a>
</div>
