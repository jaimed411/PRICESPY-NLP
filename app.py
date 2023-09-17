import streamlit as st
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import regex as re
import spacy
from stop_words import get_stop_words

# Cargar el modelo SVM previamente entrenado
model = joblib.load(r"G:\Mi unidad\1 Archivos py\4 GEEK\2 PROYECTOS\29 FINAL INMO\03 NPL DESCRIPTION\ModelSVM1.sav")

# Cargar el DataFrame con los datos
df = pd.read_csv(r"G:\Mi unidad\1 Archivos py\4 GEEK\2 PROYECTOS\29 FINAL INMO\03 NPL DESCRIPTION\falsas.csv", sep=';', encoding='latin1') 

# Cargar el modelo de lenguaje de spaCy para lematización
nlp = spacy.load("es_core_news_sm")

# Obtener las stop words en español desde scikit-learn
stop_words = get_stop_words('spanish')

# Configurar el vectorizador
vectorizer = TfidfVectorizer(max_features=5000, max_df=0.8, min_df=5, stop_words=stop_words)

# Ajustar el vectorizador a los datos de entrenamiento
X_train = vectorizer.fit_transform(df["descripcion"]).toarray()

# Interfaz de usuario de Streamlit
st.title("Detector de Fraudes")

# Agregar un área de texto para ingresar la descripción
text_input = st.text_area("Ingrese una descripción:")

if st.button("Predecir Fraude"):
    # Preprocesar el texto
    text_input = re.sub(r'[^a-zA-ZáéíóúÁÉÍÓÚñÑüÜ]', ' ', text_input)
    text_input = ' '.join(text_input.split())
    
    # Realizar lematización
    doc = nlp(text_input)
    lemmatized_text = ' '.join([token.lemma_ for token in doc])
    
    # Vectorizar el texto de entrada
    vectorized_text = vectorizer.transform([lemmatized_text]).toarray()
    
    # Realizar la predicción
    prediction = model.predict(vectorized_text)
    
    # Mostrar el resultado
    if prediction[0] == 1:
        st.error("Esta descripción parece ser fraudulenta.")
    else:
        st.success("Esta descripción parece ser legítima.")
