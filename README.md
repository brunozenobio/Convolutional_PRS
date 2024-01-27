<p align="center"><img src="https://user-images.githubusercontent.com/67664604/217914153-1eb00e25-ac08-4dfa-aaf8-53c09038f082.png"></p>

<h1 align='center'> Proyectos: Detección de imagenes de piedra papel o tijeras con redes neuronales convolucionales</h1>


<h2 align='center'>Bruno Zenobio, DATAFT16</h2>

---

## **`Tabla de Contenidos`**

- [Introducción](#introducción)
- [Desarrollo](#desarrollo)
    - [Extracción de los datos](#extracción-de-los-datos)
    - [Preparación](#preparación-de-los-datos)
    - [Rendimientos](#modelo-de-recomendación)
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

Teniendo los 3 dataset limpios, se realizó un proceso de EDA para realizar gráficos y así entender las estadísticas, encontrar valores atípicos y orientar un futuro análisis.

#### `steam_games`

- Primero se encontró la distribución de los precios a partir de un gráfico de cajas y bigotes, encontrando muchos valores atípicos. Sin embargo, considerando el contexto, no son valores necesariamente erróneos, ya que se pueden encontrar juegos de centavos de dólar y juegos de miles de dólares. Los últimos son los menos usuales.
- Se hizo un gráfico de barras con la distribución de juegos por año, incluyendo los contenidos Free. Se encontró que el año 2015 tuvo la mayor cantidad de juegos y la mayor cantidad de Free.

#### `user_reviews`

- Se realizó un gráfico de barras con la cantidad de sentimientos positivos y de estos, los que recomiendan. El resultado mostró que hubo muchos sentimientos positivos seguidos de los neutrales. Además, se observó que un porcentaje de los sentimientos positivos no recomiendan y en los sentimientos negativos, un porcentaje sí recomienda. Esto podría deberse a alguna falla en el análisis de sentimiento, sin embargo, es un porcentaje bajo.

#### `user_items`

- Para la columna playtime_forever, con un diagrama de cajas y bigotes, se analizó la distribución y se encontraron muchos valores atípicos. No obstante, a falta de un mejor análisis, no se realizará un tratamiento, ya que no necesariamente son errores; sin embargo, se debe verificar si hay algún valor de playtime_forever que para el ítem dado tenga más horas que el año de lanzamiento del ítem.
- En la verificación, se encontró que no hay ningún valor que cumpla estas condiciones, por lo que no se modificarán estos valores.
- También se calculó la dispersión entre playtime_forever y la cantidad de ítems.
- A partir de varias tablas, se graficaron los 15 juegos con más horas y los 15 desarrolladores con más horas en sus juegos.
- También se graficaron los desarrolladores con más recomendaciones positivas.

### Modelo de Recomendación

#### `Filtro Colaborativo`

- A partir de la tabla de user_reviews, se utilizaron las columnas user_id, item_id, recommend y sentiment_analysis. A partir de las dos últimas, se generó una nueva columna llamada 'rating', la cual tiene una escala entre 0 y 5. Se utilizó la técnica de Descomposición de Valor Singular (SVD) para realizar un filtro colaborativo en función de estas 3 columnas. Se utilizó GridSearch para elegir hiperparámetros óptimos; el modelo final obtuvo un RMSE de 0.85. En una escala de 0 a 5, una desviación de 0.85 es un resultado aceptable, teniendo en cuenta que el ranking podría haberse elegido de manera más óptima. EL modelo se exportó como pkl para posteriormete ser consumido por la API a través de la función<br>

Puntos a mejorar: Se podría haber elegido otro modelo como KNNBasic. Además, el proceso de generación de ratings podría haberse optimizado. También es importante considerar que un filtro basado únicamente en estas características podría estar sesgado, ya que los usuarios tienden a opinar más sobre productos que no les gustaron de forma negativa. Para abordar este problema, se podría complementar con un perfil de usuario y explorar similitudes entre ellos.

#### `Filtro basado en Contenido`

- Usando la tabla de steam_games y tomando las columnas de géneros como dummies, NearestNeighbors, el cual se encarga de buscar los k vecinos mas cercanos. Puntos a mejorar para este filtro: usar otras columnas como desarrollador o especificaciones del juego. Además, no se pudo verificar adecuadamente la performance del modelo.

Como extra en cuanto al filtro colaborativo, al usar la tabla de revisiones de usuarios, si este no se encontraba, no podía hacer recomendaciones. Por esto, para aquellos casos, se buscaba al usuario en la tabla de ítems y se realizaba una recomendación con el filtro basado en contenido en función del ítem en el que tuviera más horas y que no estuviera en las revisiones.

En general, se obtuvieron modelos aceptables; sin embargo, con un análisis más profundo, podrían haberse obtenido mejores resultados. Algunas mejoras podrían incluir la alteración ponderada, la fusión de resultados o incluso un **modelo híbrido complejo**.

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
