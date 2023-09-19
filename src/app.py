import streamlit as st
import joblib
import numpy as np
import pandas as pd
import regex as re
from wordcloud import WordCloud
from sklearn.svm import SVC
import nltk
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Descargar recursos de NLTK..
nltk.download('popular')
nltk.download("omw-1.4")
nltk.download("stopwords")
nltk.download("wordnet")

model = "./models/ModelSVM0.sav"
vectorizer = "./data/VECTOR1.pkl"

# Cargar el modelo SVM previamente entrenado
model = joblib.load(model)

# Cargar el vectorizador TF-IDF previamente entrenado
vectorizer = joblib.load(vectorizer)


# Función de preprocesamiento de texto en español.
def preprocess_text_spanish(text):
    text = re.sub(r'[^a-záéíóúüñ ]', " ", text, flags=re.IGNORECASE)
    text = re.sub(r'\s+', " ", text.lower())
    text = re.sub("&lt;/?.*?&gt;", " &lt;&gt; ", text)
    return text.split()

# Inicializar el lematizador y las palabras vacías en español
lemmatizer = WordNetLemmatizer()
stop_words = stopwords.words("spanish")

# Función para lematizar texto
def lemmatize_text(words, lemmatizer=lemmatizer):
    tokens = [lemmatizer.lemmatize(word) for word in words]
    tokens = [word for word in tokens if word not in stop_words]
    tokens = [word for word in tokens if len(word) > 3]
    return tokens

# Función para generar la nube de palabras
def generate_wordcloud(text, max_words=500, min_font_size=12):
    wordcloud = WordCloud(width=400, height=400, background_color="black", max_words=max_words, min_font_size=min_font_size, random_state=82) \
        .generate(text)
    return wordcloud

# Cargar el DataFrame con los datos
df = pd.read_csv(r"/workspaces/PROYECTO-FINAL-NLP/data/falsas.csv", sep=';', encoding='latin1') 

# Estilo CSS para centrar el contenido
st.markdown("""
<style>
.centered {
    display: flex;
    justify-content: center;
    align-items: center;
    flex-direction: column;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

# Título de la aplicación centrado
st.title("Detector de Anuncios Fraudulentos")

# Ingrese una descripción de anuncio centrada
with st.markdown("<div class='centered'>", unsafe_allow_html=True):
    nuevos_datos = st.text_area("Ingrese una descripción de anuncio:", key="text_area")

# Preprocesamiento de texto y predicción
if st.button("Verificar Anuncio"):
    if nuevos_datos:
        # Preprocesamiento del texto
        nuevos_datos = preprocess_text_spanish(nuevos_datos)
        nuevos_datos = lemmatize_text(nuevos_datos)
        nuevos_datos = " ".join(nuevos_datos)
        
        # Vectorización del texto
        vectorized_nuevos_datos = vectorizer.transform([nuevos_datos]).toarray()
        
        # Predicción
        prediction = model.predict(vectorized_nuevos_datos)
        
        # Mostrar el resultado centrado
        with st.markdown("<div class='centered'>", unsafe_allow_html=True):
            if prediction[0] == 0:
                st.success("El anuncio parece ser legítimo.")
            else:
                st.error("El anuncio parece ser fraudulento.")
        
        # Nube de palabras centrada
        with st.markdown("<div class='centered'>", unsafe_allow_html=True):
            st.header("Nube de Palabras")
            descripciones_spam_0 = df[df['spam'] == 0]['descripcion']
            descripciones_spam_1 = df[df['spam'] == 1]['descripcion']
            
            # Dividir el espacio en dos columnas
            col1, col2 = st.columns(2)
            
            # Plotea la primera nube de palabras en la primera columna
            with col1:
                st.markdown("<h4 style='font-size: 16px;'>Nube de Palabras - Anuncios no fraudulentos</h4>", unsafe_allow_html=True)
                wordcloud_spam_0 = generate_wordcloud(str(descripciones_spam_0))
                plt.figure(figsize=(8, 8))
                plt.imshow(wordcloud_spam_0)
                plt.axis("off")
                st.pyplot(plt.gcf())
            
            # Plotea la segunda nube de palabras en la segunda columna
            with col2:
                st.markdown("<h4 style='font-size: 16px;'>Nube de Palabras - Anuncios fraudulentos</h4>", unsafe_allow_html=True)
                wordcloud_spam_1 = generate_wordcloud(str(descripciones_spam_1))
                plt.figure(figsize=(8, 8))
                plt.imshow(wordcloud_spam_1)
                plt.axis("off")
                st.pyplot(plt.gcf())
    else:
        st.warning("Por favor, ingrese una descripción de anuncio antes de verificar.")
