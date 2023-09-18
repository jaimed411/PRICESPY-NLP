import streamlit as st
import joblib

# Cargar el modelo SVM previamente entrenado
model = joblib.load("ModelSVM0.sav")

# Cargar el vectorizador TF-IDF previamente entrenado
vectorizer = joblib.load("VECTOR1.pkl")

# Función para predecir si un texto es un anuncio fraudulento
def predict_fraud(text):
    # Preprocesar el texto utilizando el vectorizador TF-IDF
    text_vectorized = vectorizer.transform([text])
    
    # Realizar la predicción utilizando el modelo SVM
    prediction = model.predict(text_vectorized)
    
    return prediction[0]

# Configurar la aplicación Streamlit
st.title("Detector de Anuncios Fraudulentos")

# Caja de texto para ingresar el texto del anuncio
user_input = st.text_area("Ingresa el texto del anuncio:")

# Botón para realizar la predicción
if st.button("Ver"):
    if user_input:
        prediction = predict_fraud(user_input)
        if prediction == 1:
            st.error("Este es un anuncio fraudulento.")
        else:
            st.success("Este no es un anuncio fraudulento.")
    else:
        st.warning("Por favor, ingresa un texto antes de presionar el botón.")
