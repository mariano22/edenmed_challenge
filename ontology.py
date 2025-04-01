PROMPT = """
Transformar una descripción en una ontología. La ontología es un conjunto de líneas que condensan la descripción. Cada línea describe una entidad. Cada línea tiene el siguiente format:
entidad.sub_entidad.sub_sub_entidad. ... = valor
El operador '.' (punto) sirve para referirse a una subentidad de la entidad padre. Se peude aplicar sucesivas veces. Una vez que se tiene el grado de especificidad deseado se le asigna un valor.
Los valores solo pueden ser booleanos (True o False) o números (como 0.55 o 100). Cualquier unidad de medida u otra cosa pasa a formar parte de la entidad o subentidad a describir.
Devolver solo la ontología, nada de texto adicional.

DESCRIPCION EJEMPLO:
La arteria pulmonar de 16mm. (normal hasta 17mm).La región parahiliar sin ensanchamientos que sugieran alteraciones. El mediastino superior de aspecto normal. La silueta cardiaca con eje aparentemente normal, de perfiles y contornos conservados con un índice cardiotorácico de .42 (normal hasta .5) Las estructuras vasculares, vena cava superior, aorta y arterias pulmonares sin alteraciones.

ONTOLOGÍA RESULTADO EJEMPLO:
arteria_pulmonar.mm = 16
region_parahiliar.ensanchamientos = False
rmediastino_superior.normal = True
silueta_cardiaca.normal = True
indice_cardiotorácico = 0.42
estructuras_vasculares.alteraciones = False
vena_cava_superior.alteraciones = False
aorta.alteraciones = False
arterias pulmonares.alteraciones = False

DESCRIPCION A PROCESAR:
{report}

ONTOLOGÍA RESULTADO:
""".strip()

def prompt_ontology_generator(html_report):
    PROMPT.format(report=html_report)