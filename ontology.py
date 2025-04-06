import os
import argparse

from tqdm import tqdm
import pickle
from openai import OpenAI

import db

PROMPT = """
Transformar una descripción en una ontología. La ontología es un conjunto de líneas que condensan la descripción. Cada línea describe una entidad. Cada línea tiene el siguiente format:
entidad.sub_entidad.sub_sub_entidad. ... = valor
El operador '.' (punto) sirve para referirse a una subentidad de la entidad padre. Se puede aplicar sucesivas veces. Una vez que se tiene el grado de especificidad deseado se le asigna un valor.
Los valores solo pueden ser booleanos (True o False) o números (como 0.55 o 100). Cualquier unidad de medida u otra cosa pasa a formar parte de la entidad o subentidad a describir.
La ontología está separada en dos secciones. La primer sección tiene una línea que dice "NORMAL". Esta sección describe los campos normales. La segunda sección tiene una línea que dice "ANORMAL". Esta sección describe los campos de la ontología con valores atípicos, anormales, que son signos de alguna malformación, enfermedad o accidente.
Devolver solo la ontología y las dos secciones, nada de texto adicional.

DESCRIPCION EJEMPLO:
La arteria pulmonar de 16mm. (normal hasta 17mm).La región parahiliar sin ensanchamientos que sugieran alteraciones. El mediastino superior de aspecto normal. La silueta cardiaca con eje aparentemente normal, de perfiles y contornos conservados con un índice cardiotorácico de .42 (normal hasta .5) Las estructuras vasculares, vena cava superior, aorta y arterias pulmonares con alteraciones visibles.
Columna lumbar con cambios osteodegenerativos.
Escoliosis lumbar izquierda

ONTOLOGÍA RESULTADO EJEMPLO:
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

DESCRIPCION A PROCESAR:
{report}

ONTOLOGÍA RESULTADO:
""".strip()

ONTOLOGY_FILE = 'data/ontology.pkl'

def prompt_ontology_generator(html_report):
    PROMPT.format(report=html_report)

def calculate_ontology(df):
    ontologies_strs = []
    n = len(df)
    client = OpenAI()
    for i in tqdm(range(len(ontologies_strs),n)):
        row = df.iloc[i]
        if not isinstance(row.report_values, list):
            ontologies_strs.append((i,None))
            continue
        html_report = '\n'.join(row.report_values)
        message = PROMPT.format(report=html_report)
        #display(HTML(html_report))
        response = client.responses.create(
            model="gpt-4o",
            input=message
        )
        ontologies_strs.append((i,response.output_text))
    return ontologies_strs

def get_ontology(df):
    if os.path.exists(ONTOLOGY_FILE):
        with open(ONTOLOGY_FILE, 'rb') as file:
            ontologies_strs = pickle.load(file)
        print("Ontology loaded from file.")
    else:
        ontologies_strs = calculate_ontology(df)        
        with open(ONTOLOGY_FILE, 'wb') as file:
            pickle.dump(ontologies_strs, file)
        print("Ontology calculated and saved.")
    
    # Parse ontology
    ontologies = []
    indexes = []
    for i,ontology_text in ontologies_strs:
        o = parse_ontology(ontology_text)
        ontologies.append( o )
        indexes.append(i)
    assert indexes == list(range(len(ontologies)))
    
    return ontologies

def parse_ontology(text):
    if not isinstance(text,str):
        return text
    lines = text.split('\n')
    caracteristicas = [[], []]
    caracteristicas_tipo = 0
    for line in lines:
        if line == 'ANORMAL':
            caracteristicas_tipo = 1
        if line.count('=')==1:
            key,value = line.split('=')
            key = key.strip()
            p = value.find('#')
            if p>0:
                value = value[:p]
            value = value.strip()
            if value in ['True', 'False']:
                value = bool(value)
            else:
                try:
                    value = float(value)
                except Exception as e:
                    print(f"Error parsing: {line}")
                    pass
            caracteristicas[caracteristicas_tipo].append((key,value))
    return caracteristicas

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Calculate the onyology of the reports")
    parser.add_argument('--api_key', type=str, required=True, help='OpenAI API key')
    args = parser.parse_args()
    os.environ["OPENAI_API_KEY"] = args.api_key
    df = db.create_dataset()
    get_ontology(df)