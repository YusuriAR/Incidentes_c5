# Datos de accidentes viales en la CDMX

Análisis de accidentes viales en la CDMX reportados por el Centro de Comando, Control, Cómputo, Comunicaciones y Contacto Ciudadano de la CDMX (C5) desde 2014 y actualizdos mensualmente.

Los datos se encuentras disponibles [en el portal de Datos Abiertos Ciudad de México](https://datos.cdmx.gob.mx/explore/dataset/incidentes-viales-c5/information/?disjunctive.incidente_c4).

El equipo de Ciencia de Datos que ejecutó el proyecto son:
- Yusuri Arciga (190063)
- Diego Villegas (197902)
- Yedam Fortiz (119523)

En este repositorio se encuentra el análisis para la clasificación de llamadas que pueden ser falsas recibidas en el C5. El achivo principal donde recopilamos los resultados es el jupyter notebook Proyecto2 en el cual se nutre con distintas funciones de modelado en python en las diversas carpetas asociadas a este repositorio.

Qué infraestructura es necesaria para correr su código:

Los requerimientos para correr el proyecto se encuentran en el archivo `requirements.txt`.


Qué infraestructura se necesita para poder correr su código? (características de la máquina):

Sistema operativo: Ubuntu 20.04.1 LTS

Procesador: Intel® Core™ i7-1065G7 CPU @ 1.30GHz × 8 

64-bits

RAM: 12gb


Cómo correr su código:

Para replicar los resultados del análisis, es necesario crear la carpeta `Data` y depositar el archivo `incidentes-viales-c5.csv`.
Se abre el notebook de Proyecto2 que se encuentra en la carpeta de notebooks y se corren las celdas para obtener los resultados del modelo.
En el proyecto_1.py se pueden cargar todos los pickles.


## Evaluación de modelo

Calculamos una K con base en el numero de incidentes reportados promedio por dia y las ambulancias que contamos; dicho valor ronda alrededor de 3.35%.
Para los valores de cobertura y de eficiencia en nuestra grafica podemos ver un comportamiento constante para la mayoría de nuestros valores en k, esto se debe al thereshold en cada corte.

Con k@3.35%:
  Eficiencia@k: 25%
  Cobertura@k: 73%

Por lo tanto usamos el corte en 3.35% lo que genera una eficiencia del 25%. El numero de 'TP' es bajo respecto al numero de 'TP' mas los 'FP'; es decir, tenemos muchos casos 'FP'.

Mientras que con el corte de 3.35% se tiene una cobertura de 73%, por lo que el numero de 'TP' es considerablemente mas alto que los 'FN'.

Un sistema con alto nivel de cobertura pero eficencia baja, regresar muchos resultados pero la mayoria de ellos con la etiqueta equivocada cuando se comparan con los datos de prueba.

En nuestro caso, lo que buscamos es tener un nivel de eficiencia alto ya que contamos con pocos recuersos para poder atender las llamadas de ayuda; por lo que el nivel de eficiencia de la prueba es malo ya que no estamos enfocando los pocos recursos que tenemos para dar apoyo en casos verdaderos.

## Sesgo y Equidad

Dado que nuestro modelo está relacionado con una acción referente a un castigo consideramos las siguientes metricas:

  - PPR Predicted Positive Rate: Nos interesa para saber cuántas predijimos como falsas y realmente lo eran.
  
  - FPR False Positive Rate: Nos interesa para saber las que nos dice son falsas cuando realmente son verdaderas nos interesa para mandar la ambulancia a quién la necesita.
  
  - FDR False Discovery Rate: Nos interesa para saber fracción de personas que realmente necesitan la ambulancia y fueron predichos como falsos.
 
### Sesgo

En PPR Predicted Positive Rate podemos ver que nuestro modelo le está diciendo 1.5 veces más a la delegación Iztapalapa que las llamadas que se reciben son falsas.


### Equidad

En PPR (Predicted Positive Rate): Tenemos que es injusto en la mayoría de las delegaciones si tomamos como referencia a la delegación Gustavo A. Madero, las únicas dos que no tienen son Miguel Hidalgo y Cuauhtémoc. Esto nos indica que las ambulancias que se envían en éstas delegaciones realmente lo necesitaban.
