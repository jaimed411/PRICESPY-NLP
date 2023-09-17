import streamlit as st
import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import regex as re
from wordcloud import WordCloud
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import nltk
import matplotlib.pyplot as plt  # Importar correctamente pyplot
import nltk
nltk.download('popular')
# Cargar el modelo SVM previamente entrenado
model = joblib.load(r"G:\Mi unidad\1 Archivos py\4 GEEK\2 PROYECTOS\29 FINAL INMO\03 NPL DESCRIPTION\ModelSVM1.sav")

# Cargar el vectorizador TF-IDF previamente entrenado
vectorizer = joblib.load(r"G:\Mi unidad\1 Archivos py\4 GEEK\2 PROYECTOS\29 FINAL INMO\03 NPL DESCRIPTION\VECTOR1.pkl")

# Función de preprocesamiento de texto en español
def preprocess_text_spanish(text):
    text = re.sub(r'[^a-záéíóúüñ ]', " ", text, flags=re.IGNORECASE)
    text = re.sub(r'\s+', " ", text.lower())
    text = re.sub("&lt;/?.*?&gt;", " &lt;&gt; ", text)
    return text.split()

# Función para lematizar texto
nltk.download("omw-1.4")  # Descargar el recurso para lematización en español
nltk.download("stopwords")  # Descargar la lista de palabras vacías en español

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download("wordnet")
lemmatizer = WordNetLemmatizer()

# Cambia "Spanish" a "spanish" en la siguiente línea para cargar las palabras vacías en español
stop_words = stopwords.words("spanish")

def lemmatize_text(words, lemmatizer=lemmatizer):
    tokens = [lemmatizer.lemmatize(word) for word in words]
    tokens = [word for word in tokens if word not in stop_words]
    tokens = [word for word in tokens if len(word) > 3]
    return tokens

# Cargar el DataFrame con los datos
df = pd.read_csv(r"G:\Mi unidad\1 Archivos py\4 GEEK\2 PROYECTOS\29 FINAL INMO\03 NPL DESCRIPTION\falsas.csv", sep=';', encoding='latin1') 

# Título de la aplicación
st.title("Detector de Anuncios Fraudulentos")

# Sidebar
st.sidebar.header("Configuración")
nuevos_datos = st.sidebar.text_area("Ingrese una descripción de anuncio:")

# Preprocesamiento de texto y predicción
if nuevos_datos:
    # Preprocesamiento del texto
    nuevos_datos = preprocess_text_spanish(nuevos_datos)
    nuevos_datos = lemmatize_text(nuevos_datos)
    nuevos_datos = " ".join(nuevos_datos)
    
    # Vectorización del texto
    vectorized_nuevos_datos = vectorizer.transform([nuevos_datos]).toarray()
    
    # Predicción
    prediction = model.predict(vectorized_nuevos_datos)
    
    # Mostrar el resultado
    if prediction[0] == 0:
        st.success("El anuncio parece ser legítimo.")
    else:
        st.error("El anuncio parece ser fraudulento.")

# Nube de palabras
st.header("Nube de Palabras")
descripciones_spam_0 = df[df['spam'] == 0]['descripcion']
descripciones_spam_1 = df[df['spam'] == 1]['descripcion']
wordcloud_spam_0 = WordCloud(width=800, height=800, background_color="black", max_words=1000, min_font_size=40, random_state=82) \
    .generate(str(descripciones_spam_0))
wordcloud_spam_1 = WordCloud(width=800, height=800, background_color="black", max_words=1000, min_font_size=40, random_state=82) \
    .generate(str(descripciones_spam_1))

# Plotea la primera nube de palabras
st.subheader("Nube de Palabras - Posibles anuncios no fraudulentos")
plt.figure(figsize=(12, 6))
plt.imshow(wordcloud_spam_0)
plt.axis("off")
st.pyplot()

# Plotea la segunda nube de palabras
st.subheader("Nube de Palabras - Posibles anuncios fraudulentos")
plt.figure(figsize=(12, 6))
plt.imshow(wordcloud_spam_1)
plt.axis("off")
st.pyplot()

# Información adicional
st.sidebar.markdown("### Información Adicional")
st.sidebar.info(
    "Este es un detector de anuncios fraudulentos que utiliza un modelo SVM entrenado previamente con TF-IDF. "
    "Ingrese una descripción de anuncio en el cuadro de texto de la barra lateral para obtener una predicción sobre "
    "si el anuncio es legítimo o fraudulento."
)
