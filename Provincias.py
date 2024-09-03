import streamlit as st
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
import numpy as np
from scipy.spatial.distance import cosine
from sklearn.metrics import jaccard_score
from scipy.spatial.distance import pdist, squareform
from sklearn.manifold import MDS
import matplotlib.pyplot as plt
import seaborn as sns
import re
@st.cache_data
def load_dat():
    datos = pd.read_csv('prenoa.csv')
    return datos
# Cargar y procesar los datos en una función con caché
@st.cache_data
def load_min():
    datos = pd.read_csv('prenoa.csv')
    pro_por_tem = datos.groupby(['Clasificador A - Temática','Jurisdicción','Ministerio/ Organismo descentralizado'])[['Nombre Programa']].count()
    pro_por_tem.reset_index(inplace=True)
    return pro_por_tem

@st.cache_data
def load_data():
    datos = pd.read_csv('prenoa.csv')
    pro_por_tem = datos.groupby(['Clasificador A - Temática','Jurisdicción'])[['Nombre Programa']].count()
    pro_por_tem.reset_index(inplace=True)
    return pro_por_tem

@st.cache_data
def load_datos():
    datos = pd.read_csv('prenoa.csv')
    pro_por_tempo = datos.groupby('Clasificador A - Temática')[['Nombre Programa']].count()
    pro_por_tempo.reset_index(inplace=True)
    return pro_por_tempo

def eliminar_numero_caracter(texto):
    # Usamos una expresión regular para encontrar un patrón de uno o más dígitos seguido de un carácter especial al inicio del texto
    resultado = re.sub(r'^\d+[^\w\s]', '', texto)
    return resultado

# Llamar a la función para cargar los datos procesados
pro_por_tem = load_data()
pro_por_tempo= load_datos()
pro_por_min= load_min()
# Lista de provincias disponibles
provincias = pro_por_tem['Jurisdicción'].unique().tolist()
provincias.insert(0, 'Todas')  # Añadir la opción 'Todas' al inicio

# Crear el menú en Streamlit para seleccionar la provincia
provincia_seleccionada = st.sidebar.selectbox('Selecciona la provincia', provincias)

# Filtrar los datos según la provincia seleccionada
if provincia_seleccionada != 'Todas':
    lista_ministerios_provincia=pro_por_min[pro_por_min['Jurisdicción'] == provincia_seleccionada]['Ministerio/ Organismo descentralizado'].unique().tolist()
    lista_ministerios_provincia.insert(0, 'Todos los organismos')
    ministerio_seleccionado= st.sidebar.selectbox("Selecciona un ministerio: ",lista_ministerios_provincia )
    if ministerio_seleccionado != 'Todos los organismos':
        datos_filtrados =pro_por_min[(pro_por_min['Jurisdicción'] == provincia_seleccionada) & ( pro_por_min['Ministerio/ Organismo descentralizado']==ministerio_seleccionado)].sort_values(by=['Nombre Programa'],ascending=False)
        datos_filtrados['Clasificador A - Temática'] = datos_filtrados['Clasificador A - Temática'].apply(eliminar_numero_caracter)
        df=load_dat()
        datos_fil= df[(df['Jurisdicción'] == provincia_seleccionada) & (
                    df['Ministerio/ Organismo descentralizado'] == ministerio_seleccionado)][['Nombre Programa','Objetivo general',
       'Objetivos específicos', 'Normativa', 'Población destinataria',
       'Requisitos de Accesibilidad', 'Criterios de elegibilidad',
       '¿Cuál es el alcance del programa?', 'Prestación']]
    else:
        datos_filtrados = pro_por_tem[pro_por_tem['Jurisdicción'] == provincia_seleccionada].sort_values(by=['Nombre Programa'],ascending=False)
        datos_filtrados['Clasificador A - Temática'] = datos_filtrados['Clasificador A - Temática'].apply(eliminar_numero_caracter)
else:
    ministerio_seleccionado ='Todos los organismos'
    datos_filtrados = pro_por_tempo.sort_values(by=['Nombre Programa'],ascending=False)
    datos_filtrados['Clasificador A - Temática']=datos_filtrados['Clasificador A - Temática'].apply(eliminar_numero_caracter)
st.markdown("""
    <h1 style='text-align: center; font-size: 30px; font-weight: bold;'>
        Programas Sociales Provincias 
    </h1>
    """, unsafe_allow_html=True)

# Crear la figura de Matplotlib explícitamente
sns.set_style('darkgrid')
fig, ax = plt.subplots(figsize=(15, 9))
barplot = sns.barplot(x='Clasificador A - Temática', y='Nombre Programa', data=datos_filtrados, alpha=0.7,ax=ax, edgecolor='none', errorbar=None)

# Añadir etiquetas y título
plt.title(f'Cantidad de Programas cargados por Provincia ({eliminar_numero_caracter(provincia_seleccionada)}) {ministerio_seleccionado}', fontsize=13)
plt.xlabel('Provincia', fontsize=11)
plt.ylabel('Cantidad de Programas', fontsize=11)

# Girar las etiquetas del eje x
plt.xticks(rotation=45, ha='right', fontsize=9)

plt.yticks(fontsize=9)

# Añadir etiquetas de datos en las barras
for p in barplot.patches:
    barplot.annotate(format(p.get_height(), '.1f'),
                     (p.get_x() + p.get_width() / 2., p.get_height()),
                     ha='center', va='center',
                     xytext=(0, 9),  # Distancia del texto a la barra
                     textcoords='offset points',
                     fontsize=9)  # Tamaño de la fuente más pequeño

# Mejorar el diseño del gráfico
sns.despine(left=True, bottom=True)

# Mostrar el gráfico en Streamlit
st.pyplot(fig)
if provincia_seleccionada != 'Todas'and ministerio_seleccionado != 'Todos los organismos':
    st.sidebar.dataframe(datos_fil)

