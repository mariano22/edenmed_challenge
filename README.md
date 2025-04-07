# EdenMed Segmentation Challenge

![Python](https://img.shields.io/badge/python-3.10-blue.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

# Uso rápido

#### Instalación

Instalar las dependencias con:
```
pip install -r requirements.txt
```

#### Segmentar una imágen

Se puede utilizar el modelo de segmentación en una imágen con el comando:
```
python segmentation.py --image data/test_classifier/a.jpg --model models/unet-6v.pt --out mask.jpg
```

> Segmentation finished!
> Mask was generated: mask.jpg

#### Determinar si la imágen corresponde a un torax

Para previamente determinar si la imágen contiene un torax se puede usar:

```
python torax_detection.py --image data/test_classifier/a.jpg --model models/torax_detector_model.pth
```
Output (clase y probabildiad respectiva):
> class=torax
> confidence=0.84

#### Solución al dataset suministrado

El dataset suministrado (`data/challenge_data`) se dividió en imágenes de tórax e imágenes que no correspondían a tórax (`data/binary_dataset`) utilizando el clasificador (`torax_detection.py`) y las imágenes respectiva a tórax se segmentaron utilizando el modelo de segmentación mencionado (`segmentation.py`) produciéndo finalmente las máscaras que se incluyeron en `data/binary_dataset_torax_masks`.

# Informe

Las siguientes secciones incluyen la guía paso a paso de cómo se ha resuelto el challenge.

## Data exploration

En la base de datos suministrada hay 3 tablas:

> Table: study_patient_data
>  Columns: study_id, body_parts, modalities, patient_age, patient_gender
> Rows: 549

Contiene información sobre el estudio en particular (tipo de modalidad, sección estudiada, datos del paciente).

> Table: study_instances
>  Columns: study_id, instance_id, measurement_data, file_name
>  Rows: 460

Para cada estudio, contiene una serie de datos de medición e imágenes asosciadas.
Nota: para cada `study_id` se encontró un único `file_name` asosciado.

> Table: study_report
>  Columns: study_id, report_id, report_value
>  Rows: 555
>

Para cada estudio, contiene un reporte en HTML escrito en lenguaje natural español.

![Tablas presentes en la base de datos](./images/tables.png)

**Otras Observaciones:**
- `study_patient_data` contiene una única fila por `study_id`.
- `study_instances` contiene varias instancias por `study_id`, pero una única imagen respectiva `file_name`.
- `study_id` contiene varios reportes por `study_id`.
- Existen `sutdy_id` que no tienen imágenes asosciadas (esto se desprende simplemente de ver la cantidad de filas).
- Existen `file_name` para los cuales no existe una imagen jpeg respectiva.

Todas  estas observaciones pueden corroborarse ejecutando `python db.py` que correrá estos chequeos.

### Exploración de características demográficas de los pacientes

Se ha calculado:
- La composición de los estudios por sexo.
- La composición de los estudios por edad del paciente.
- La composición de los estudios por modalidad (CT, CX, DT, DX).
- La distribución de los CTR.
**Nota**: el código que genera los gráficos está en la notebook: `notebook.ipynb`

![Composición de los estudios por sexo.](./images/sex_composition.png)

![Composición de los estudios por edad.](./images/age.png)
![Composición de los estudios por modalidad.](./images/modalities_composition.png)
![Distribución CTR](./images/ctr.png)

### Propuesta para el análisis de los reportes médicos

Para el análisis de los reportes médicos se propone generar una ontología para cada reporte.
La ontología es una secuencia de pares (clave, valor) extrayendo todos los datos estructurados del reporte que se encuentren.
Se realiza utilizando un LLM. Actualmente se usa gpt-4o de OpenAI aunque fácilmente se podría adaptar para usar otra interfaz.
Además de eso, clasifica en "NORMAL" y "ANORMAL" cada observación.

Por ejemplo el reporte:

````{verbatim}
La arteria pulmonar de 16mm. (normal hasta 17mm).La región parahiliar sin ensanchamientos que sugieran alteraciones. El mediastino superior de aspecto normal. La silueta cardiaca con eje aparentemente normal, de perfiles y contornos conservados con un índice cardiotorácico de .42 (normal hasta .5) Las estructuras vasculares, vena cava superior, aorta y arterias pulmonares sin alteraciones.
````

Generaría la siguiente ontología:
```
NORMAL
arteria_pulmonar.mm = 16
region_parahiliar.ensanchamientos = False
rmediastino_superior.normal = True
silueta_cardiaca.normal = True
indice_cardiotorácico = 0.42
ANORMAL
estructuras_vasculares.alteraciones = True
vena_cava_superior.alteraciones = True
aorta.alteraciones = True
arterias pulmonares.alteraciones = True
columna_lumbar.cambios_osteodegenerativos = True
columna_lumbar.escoliosis_izquierda = True
```

El prompt para generar dichas ontologías se encuentra en `ontology.py`

#### Analizando los reportes

La ventaja de este abordaje es que permite realizar análisis de los reportes con simples reglas, recorriendo la ontología generada y las llamadas a la API de LLM (lento y costoso) se hace una sóla vez. 

En las siguientes subsecciones se encuentran algunos análisis que se pueden realizar (el código respectivo para calcularlo está en `notebook.ipynb`).

La ontología de cada estudio está guardada en `data/ontology.pkl` y el código usado para calcularla está en `ontology.py`

##### Histograma de distribución de Ángulo de Ferguson

![IHistograma de distribución de Ángulo de Ferguson](./images/ferguson_angles.png)

##### Casos diagnosticados de escoleosis

```
14 casos con escoliosis: 
- 003b88ed-bdd5-4bac-89b2-c170d9418d75
- 0403dfdd-9ac4-4d6b-86ab-fb3647e90891
- 082ca1b1-a9a2-4c77-9a55-64f06c920f4b
- 0df07efa-1e39-4c57-96cf-eee4a2e16eff
- 1ec28fe0-2868-494b-b129-21caf9a01927
- 21e23158-711c-478a-8429-0ac9caa00484
- 4aa65847-6f5f-4559-8bd0-7482ed01f097
- 5c016336-0d4e-4b7f-aa33-a518f00b0bf4
- 904c8271-12a2-4015-9f62-f5ea97fe2337
- a1e11c71-583e-4620-90eb-ebe901676694
- a225cdb1-ec06-428f-8aa2-f19bf892000a
- b43aeabc-f687-4aa6-976d-489a9ecb8ed2
- daf33fc7-870a-424b-a169-d538f21d2fa2
- f97c2d0e-c063-4b02-adbb-ca67e154f2b5
```

## Construcción del dataset

#### Unificando las tablas

Tener 3 tablas normalizadas es útil para el funcionamiento de un sistema transaccional. Pero a la hora de operar con datos de forma analítica lo mejor es tener toda la información en **una sóla tabla** (por más que no esté normalizada). Para ello se:
1. Agrupó por `study_id` las tablas `study_instances` y `study_report`. Las columnas se agregaron como listas de valores.
2. Se hizo un merge (JOIN de tablas) por `study_id`.
3. Se eliminaron filas que no contenían un `file_name` o para el cuál no existían imágenes asosciadas.

Esto se encuentra en la función `db.create_dataset`:

#### Seleccionando las imágenes de tórax

Ahora debemos seleccionar las imágenes que correspondan efectivamente a torax. Analizando los `body_parts` vemos los siguientes valores:

> {'TÓRAX', 'L SPINE', 'TORAX TORAX LATERAL', 'COLUMNA', 'BREAST', 'DEFAULT', 'LSPINE', 'SHOULDER', 'COLUMNA COLUMNA-L/S LATERAL', 'Pectoral', 'COL. LUMBAR', 'CSPINE', 'TORAX', 'PELVIS', 'TORAX TORAX PA', 'Senos Paranasales', 'NECK', 'THORAX', 'SPINE LUMBAR', 'PNS', 'L-SPINE', 'ABDOMEN', 'CHEST', 'OTHER', 'L/S-SPINE', 'HEAD', 'COLUMNA COLUMNA-L/S AP', 'ANGIO'}

Para seleccionar vamos a realizar los siguientes pasos:
1. Una primera selección naive, basada en `body_parts` razonables, para separarlos en clases `torax` y `not_torax`.
2. Limpiar a manualmente cada una de las dos carpetas. Para eso:
a. Viendo las miniaturas de las imágenes en `torax` seleccionamos las que no sean de torax y las movemos a la carpeta `not_torax_cleaned`.
b. Viendo las miniaturas de las imágenes en `not_torax` seleccionamos las que SI sean de torax y las movemos a la carpeta `torax_cleaned`.
c. Finalmente copiamos los archivos restantes en `not_torax` y `torax` a `torax_cleaned` y `not_torax_cleaned`.
**Observacion**: podemos hacer un limpiado a mano rápidamente viendo las imágenes como minuaturas por dos razones. En primer lugar nuestras clases son fácilmente identificables (podemos hacerlo viendo la miniatura). En segundo lugar, tenemos pocas imágenes (alrededor de 300).
Luego de este paso estaremos en condiciones de **entrenar un clasificador binario** lo que nos permitirá tener una selección más precisa para imágenes futuras (para las cuales, además, podemos no tener datos como `body_part`).
3. Pondremos estas dos clases en la carpeta `binary_dataset` y entranaremos un clasificador binario basado en `Resnet`.

El modelo de selección de tórax quedó guardado en `models/torax_detector_model.pth` y puede ser usado en una imagen de la siguiente manera:
```
python torax_detection.py --image data/test_classifier/a.jpg --model models/torax_detector_model.pth
```
Output (clase y probabildiad respectiva):
> class=torax
> confidence=0.84

Ejemplos de imágenes de internet clasificadas:

![Imágenes de internet clasificadas por el clasificador entrenado](./images/classifier_results.png)

## Segementando pulmones

Para la segmentación pulmonar se ha utilizado una arquitectura U-Net (ver Apéndice II: Referencias).
La misma ha sido entrenada con **Montgomery** y **Shenzhen** datasets de readiografías de torax (ver Apéndice I: Datasets).
El código que se ha utilizado es el de [Repositorio de segmentación pulmonar (GitHub)](https://github.com/IlliaOvcharenko/lung-segmentation).

Un ejemplo de una imágen del dataset superpuesta con la máscara generada por el modelo:

![Ejemplo de imágen superpuesta por su máscara de segmentación.](./images/segmentation_example.png)

Se puede utilizar el modelo de segmentación en una imágen con el comando:
```
python segmentation.py --image data/test_classifier/a.jpg --model models/unet-6v.pt --out mask.jpg
```

> Segmentation finished!
> Mask was generated: mask.jpg

El comando anterior generará una imagen blanco y negro con la máscara en el path que se le indique en el argumento --out

![Composición de los estudios por modalidad.](./images/output_mask.png)

Las imágenes de `data/binary_classifier/torax` se han procesado con el segmentador y guardado en `data/binary_classifier_torax_mask`

### Por qué elegimos esta implementación?

La razón de por qué elegimos [este repositorio](https://github.com/IlliaOvcharenko/lung-segmentation) es porque ya proveía los pesos del modelo entrenado en un dataset conocido. **Y más importante aún posee:**
- El código de entrenamiento mediante el cual podemos verificar que ha sido entrenado correctamente (con splits de train, validación y test).
- Los splits (`splits.pk`) que nos permite replicar el entrenamiento realizado.

### Entrenando nuestro propio modelo

En el archivo `segmentation_train.py` se encuentra el entrenamiento de otro modelo UNet, cuya arquitectura hemos copiado de [Pytorch-UNet](https://github.com/milesial/Pytorch-UNet).

El loop de entrenamiento lo escribimos totalmente de cerp. Y para entrenar usamos un split de los datos con una mejor metodología que la adoptada por el modelo usado en las secciones anteriores:
- Para el conjunto de desarrollo (train + validación): utilizamos *Shenzhen Dataset*
- Para el conjunto de test: utilizamos *Montgomery Dataset*

El objetivo con esta metodología es simular que el dataset en el que se va a usar es distinto (posiblemente con distintos aparatos de imágen) que el dataset de entrenamiento.

Los pesos del modelo se guardaron en `models/unet-enmed.pt`.

Las curvas costo y métricas en validación y entrenamiento se grafican a continuación:

![Training graphs](./images/training.png)

En el split de test se han obtenido las siguientes métricas: **Jaccard score - 0.5436, Dice score - 0.7022** . (comparables al modelo utilizado en las secciones anteriores).


#### Acotaciones y trabajo futuro

Algunas notas de pendientes y posibles mejores:
- Se ha utilizado un mini batch de tamaño 4. Esto es porque valores mayores no entraban en la memoria de la GPU usada (P5000). Se podría simular mini-batches más grande utilizando `accumulation_steps` durante el entrenamiento.
- El tamaño del dataset de entrenamiento es pequeño ~530 imágenes (85% del Shenzhen Dataset). Se podría aplicar data augmentation para enrobustecer el modelo. La modificación requiere ser cuidadoso de preservar la correspondencia de los píxeles entre inputs y máscaras, pero esto lo maneja todo muy bien librerías como [albumentations](https://github.com/albumentations-team/albumentations).
- Data augmentation también puede aplicarse en tiempo de inferencia (y luego hacer un merge los resultados). Esto requiere transformaciones inversibles, para lo cuál se puede utilizar la librería [ttach](https://github.com/qubvel/ttach)


# Apéndice I: Datasets

**Montgomery Dataset**  
- **Origen:** Montgomery County, Maryland, USA  
- **Número de imágenes:** ~138  

**Shenzhen Dataset**  
- **Origen:** Hospital Nº3 de Shenzhen, China  
- **Número de imágenes:** ~662

Ambos datasets se encuentran en el siguiente [archivo](https://drive.google.com/file/d/1ffbbyoPf-I3Y0iGbBahXpWqYdGd7xxQQ/view). El directorio `images` contiene las radiografías y `masks` el source of truth de las máscaras. Las máscaras tienen el mismo nombre que su respectiva imágen y se le ha adicionado el sufijo `_mask`. Los archivos del Shenzhen Dataset comienzan con `CHNCXR`.

```
📁 dataset/
├── 📁 images/
│   ├── CHNCXR_0001_0.png  
│   ├── CHNCXR_0002_0.png  
│   ├── ...  (662 archivos)
│   ├── MCUCXR_0001_0.png    
│   ├── MCUCXR_0002_0.png    
│   └── ...  (138 archivos)
└── 📁 mask/
    ├── CHNCXR_0001_0_mask.png  
    ├── CHNCXR_0002_0_mask.png  
    ├── ...  (662 archivos)
    ├── MCUCXR_0001_0_mask.png  
    ├── MCUCXR_0002_0_mask.png  
    └── ...  (138 archivos)
```

# Apéndice II: Referencias

Ronneberger, O., Fischer, P., & Brox, T. (2015). U-Net: Convolutional Networks for Biomedical Image Segmentation. *In Proceedings of the International Conference on Medical Image Computing and Computer-Assisted Intervention (MICCAI)* (pp. 234–241). Springer. https://doi.org/10.1007/978-3-319-24574-4_28
