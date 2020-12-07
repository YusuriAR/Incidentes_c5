# Datos de accidentes viales en la CDMX

Analisis de accidentes viales en la CDMX reportados por el Centro de Comando, Control, Cómputo, Comunicaciones y Contacto Ciudadano de la CDMX (C5) desde 2014 y actualizdos mensualmente.

Los datos se encuentras disponibles [en el portal de Datos Abiertos Ciudad de México](https://datos.cdmx.gob.mx/explore/dataset/incidentes-viales-c5/information/?disjunctive.incidente_c4).

El equipo de Ciencia de Datos que ejecutó el proyecto son:
- Yusuri Arciga (190063)
- Diego Villegas (197902)
- Yedam Fortiz (119523)

Los requerimientos para correr el proyecto se encuentran en el archivo `requirements.txt`.
Para replicar los resultados del analisis, es necesrio crear la carpeta `Data` y depositar el archivo `incidentes-viales-c5.csv`.

# x

Qué infraestructura es necesaria para correr su código:

# x

Qué infraestructura se necesita para poder correr su código? (características de la máquina):

# x

Cómo correr su código:

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
  *PPR Predicted Positive Rate: Nos interesa para saber cuántas predijimos como falsas y realmente lo eran.
  *FPR False Positive Rate: Nos interesa para saber las que nos dice son falsas cuando realmente son verdaderas nos interesa para mandar la ambulancia a quién la necesita.
  *FDR False Discovery Rate: Nos interesa para saber fracción de personas que realmente necesitan la ambulancia y fueron predichos como falsos.
 
# x

Sesgo

# x

Equidad
