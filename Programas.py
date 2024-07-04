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

# Cargar modelos y datos en caché
@st.cache_resource
def load_translator():
    translator_es_to_en = pipeline("translation_es_to_en", model="Helsinki-NLP/opus-mt-es-en")
    return translator_es_to_en

@st.cache_resource
def load_model():
    model = SentenceTransformer("all-mpnet-base-v2")
    return model
@st.cache_data
def load_matriz():
    matriz=pd.read_csv('matriz_de_similaridad_con_nombres.csv',index_col=0)
    return matriz
@st.cache_data
def load_data():
    datos = pd.read_csv('programas.csv')

    categorias_en = [
        "Youth and Adolescence", "Public Health", "Nutrition", "Education", "Employment and Job Training",
        "Social Inclusion", "Diversity and Equality", "Culture", "Sports", "Tourism",
        "Environment and Sustainability", "Food Security", "Addiction Prevention", "Mental Health",
        "Disability", "Chronic Disease Management", "Disease Prevention", "Animal Health",
        "Human Rights", "Community Participation", "Economic Support", "Social Development",
        "Housing and Habitability", "Legal Assistance", "Women's Empowerment", "Environmental Control",
        "Water Care", "Climate Change Protection", "Technical Training", "Health Promotion",
        "Agriculture and Rural Development", "Innovation and Technology", "Waste Management", "Infrastructure",
        "Transport", "Safety and Justice", "Entrepreneurship Support", "Industrial Development",
        "Food Quality and Safety", "Digital Inclusion", "Urban Development", "Local Economy Promotion",
        "Biodiversity Protection", "Natural Resources", "Research and Development", "Social Assistance",
        "Institutional Strengthening", "Gender Policies", "Animal Protection and Welfare",
        "Promotion of Sports and Physical Activity"
    ]
    categorias_es = [
        "Juventud y Adolescencia", "Salud Pública", "Nutrición", "Educación", "Empleo y Formación Laboral",
        "Inclusión Social", "Diversidad e Igualdad", "Cultura", "Deportes", "Turismo",
        "Ambiente y Sustentabilidad", "Seguridad Alimentaria", "Prevención de Adicciones", "Salud Mental",
        "Discapacidad", "Atención a Enfermedades Crónicas", "Prevención de Enfermedades", "Salud Animal",
        "Derechos Humanos", "Participación Comunitaria", "Apoyo Económico", "Desarrollo Social",
        "Vivienda y Habitabilidad", "Asistencia Legal", "Empoderamiento de Mujeres", "Control Ambiental",
        "Cuidado del Agua", "Protección contra el Cambio Climático", "Capacitación Técnica", "Promoción de la Salud",
        "Agricultura y Desarrollo Rural", "Innovación y Tecnología", "Gestión de Residuos", "Infraestructura",
        "Transporte", "Seguridad y Justicia", "Apoyo a Emprendedores", "Desarrollo Industrial",
        "Calidad e Inocuidad Alimentaria", "Inclusión Digital", "Desarrollo Urbano", "Fomento de la Economía Local",
        "Protección de la Biodiversidad", "Recursos Naturales", "Investigación y Desarrollo", "Asistencia Social",
        "Fortalecimiento Institucional", "Políticas de Género", "Protección y Bienestar Animal",
        "Promoción del Deporte y la Actividad Física"
    ]

    categ = [str.lower(c) for c in categorias_en]

    return categ, datos, categorias_es

# Clasificador
@st.cache_resource
def load_clasifier():
    classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
    return classifier

# Funcion para traducir texto
def translate_es_to_en(text):
    translator_es_to_en=load_translator()
    translation = translator_es_to_en(text)
    return translation[0]['translation_text']


# Funcion para separar parrafos de un texto largo
def separar_texto_lineas(texto, sep='\n\n'):
    return str.split(texto, sep=sep)


# Funcion para agregar variables binarias a un dataframe segun el programa coincida con la categoria
def etiquetas(texto, categoria):
    for parrafo in texto:
        if len(parrafo) > 0:
            cla = load_clasifier(str.lower(parrafo),
                             candidate_labels=[str.lower(categoria)], )
            if float(cla['scores'][0]) > 0.9:
                print(cla['scores'][0])
                return 1

    return 0
#Funcion para calcular similaridades
def similaridades(pal='turismo',categ=load_data()[0],n=50):
    model=load_model()
    categorias_es=load_data()[2]
    datos=load_data()[1]
    palabra1=str.lower(translate_es_to_en(pal))
    #palabra2=translate_es_to_en(pal)
    embeddings = model.encode(categ)
    embeddings_pal = model.encode([palabra1])
    similarities = model.similarity(embeddings, embeddings_pal)
    sim=similarities.reshape(n,)
    dfsim = {'Categoria':categorias_es , 'Sim': sim}
    dfsim=pd.DataFrame(dfsim).sort_values(by='Sim',ascending=False)
    df_max_pro=pd.DataFrame(datos.loc[datos[categorias_es[sim.argmax()]]==1,['Programas','Descripcion']]).reset_index(drop=True)
    return [df_max_pro,dfsim]


# Funcion para calcular la matriz de distancias y similaridades
def matriz_distancias_similaridades(datos):
    binary_matrix = datos.values

    # Calcular la matriz de distancia (1 - Jaccard Similarity)
    distances = pdist(binary_matrix.T, metric='jaccard')

    # Convertir las distancias a una matriz cuadrada
    distance_matrix = squareform(distances)
    return distance_matrix
    # Paso 2: Calcular las coordenadas principales usando PCoA (MDS con metric=False)
    # mds = MDS(n_components=2, dissimilarity='precomputed', metric=False)

    # Ajustar MDS a la matriz de distancia
    # principal_coordinates = mds.fit_transform(distance_matrix)

    # Crear un DataFrame con las coordenadas principales
    # pcoa_df = pd.DataFrame(data=principal_coordinates, columns=['PCo1', 'PCo2'], index=df.columns)


def realizar_suma_ponderada(datos, pesos, seleccionadas):
    df = load_data()[1]

    # Lista de columnas a sumar ponderadamente
    columnas_a_sumar = seleccionadas

    # Lista de pesos correspondientes a cada columna
    pesos = pesos

    # Verificar que la longitud de las listas coincida
    if len(columnas_a_sumar) != len(pesos):
        raise ValueError("La longitud de columnas_a_sumar y pesos debe ser la misma.")

    # Crear una nueva columna con la suma ponderada de las columnas especificadas
    df['suma_ponderada'] = sum(df[col] * peso for col, peso in zip(columnas_a_sumar, pesos))

    # Mostrar el DataFrame resultante
    return df.sort_values(by='suma_ponderada', ascending=False).loc[:, ['Programas', 'Descripcion', 'suma_ponderada']]


# Funcion para pbtener programas similares
def obtener_programas_similares(similarity_matrix, programa, top_n=10):
    # Verificar si el programa existe en la matriz
    if programa not in similarity_matrix.columns:
        raise ValueError(f"El programa '{programa}' no se encuentra en la matriz de similitud.")

    # Obtener la fila correspondiente al programa
    similaridades = similarity_matrix[programa]

    # Ordenar las similaridades en orden descendente y obtener los índices de los programas más similares
    # Excluir el propio programa usando 'iloc[1:top_n+1]' en lugar de 'iloc[:top_n]'
    programas_similares = similaridades.sort_values(ascending=False).iloc[1:top_n + 1]

    return pd.DataFrame(programas_similares)


#datos['Parrafos']=datos['Traduccion'].apply(separar_texto_lineas)
# CSS personalizado para cambiar colores de botones y texto

primaryColor = "#4CAF50"  # Verde para los botones
backgroundColor = "#FFFFFF"  # Fondo blanco
secondaryBackgroundColor = "#F0F2F6"  # Fondo de los contenedores y expander
textColor = "#000000"  # Texto negro
font = "sans serif"  # Fuente predeterminada

st.markdown("""
    <style>
    .stButton button {
        background-color: #4CAF50; /* Green */
        border: none;
        color: white;
        padding: 10px 20px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 14px;
        margin: 4px 2px;
        transition-duration: 0.4s;
        cursor: pointer;
    }
    .stButton button:hover {
        background-color: white;
        color: black;
        border: 2px solid #4CAF50;
    }
    .stTextInput > div > input {
        padding: 10px;
        font-size: 14px;
    }
    </style>
""", unsafe_allow_html=True)


# Configuración de la aplicación Streamlit
st.title("Busca un programa en la Guía de Programas Sociales de Tucumán ")

# Entrada del usuario
input_palabra = st.text_input("Introduce una palabra o frase en español", "turismo")

# Botón para calcular similaridades
if st.button("Calcular Similitudes"):
    df_programas, df_similitudes = similaridades(input_palabra)

    st.session_state['df_programas'] = df_programas
    st.session_state['df_similitudes'] = df_similitudes

    st.subheader("Programas más similares")
    st.dataframe(df_programas)

    st.subheader("Similitudes con categorías")
    st.dataframe(df_similitudes)

    st.header('Programas Seleccionados')
    st.subheader('seleccione el programa' )

    #st.dataframe(obtener_programas_similares(load_matriz,st.session_state[programa]))
 # Botón para procesar programas
if st.button("Calcular Suma Ponderada"):
    if 'df_similitudes' in st.session_state:
        datos=load_data()[1]
        dfs=st.session_state['df_similitudes']
        Cat = dfs[dfs['Sim'] > 0.5]
        seleccionadas = list(Cat['Categoria'])
        pesos = list(Cat['Sim'])
        resultado_programas = realizar_suma_ponderada(datos, pesos, seleccionadas)
        st.dataframe(resultado_programas)
    else:
        st.warning("Primero calcula las similitudes.")

if 'df_programas' in st.session_state:
    st.subheader("Selecciona un Programa")
    selected_programa = st.selectbox("Selecciona un programa para ver similares", st.session_state['df_programas']['Programas'])
if st.button("Mostrar Programas Similares"):
    st.session_state['selected_programa'] = selected_programa
    # Crear columnas para distribuir los botones horizontalmente
    #cols = st.columns(3)  # Ajusta el número de columnas según sea necesario
    #for i, programa in enumerate(st.session_state['df_programas']['Programas']):
           # with cols[i % 3]:
                #if st.button(programa):
                   # st.session_state['selected_programa']=programa
if 'selected_programa' in st.session_state:
    prosim = obtener_programas_similares(load_matriz(),st.session_state['selected_programa'])
    prosim.reset_index(names='Programas', inplace=True)
    datos=load_data()[1]
    resultados=pd.merge(prosim, datos, on='Programas').iloc[:, [0,2,1]]
    st.dataframe(resultados)



