import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt

# Título do app
st.set_page_config(page_title='Brain Tumor Classifier', layout='centered')
st.title('🧠 Classificador de Tumores Cerebrais por MRI')
st.write('Este modelo de Deep Learning prevê **4 classes** a partir de imagens de ressonância magnética:')
st.markdown('''
- 🟤 Glioma  
- 🟡 Meningioma  
- 🟢 Sem Tumor  
- 🟣 Pituitary
''')

# Carregar modelo
@st.cache_resource
def carregar_modelo():
    return tf.keras.models.load_model('modelo.treinado.h5')

modelo = carregar_modelo()

# Labels
classes = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']

# Preprocessamento
def preprocessar_imagem(imagem):
    imagem = imagem.resize((224, 224)).convert('RGB')
    img_array = np.array(imagem) / 255.0
    return np.expand_dims(img_array, axis=0)

# Upload de imagem
uploaded_file = st.file_uploader('📤 Envie uma imagem de ressonância cerebral (JPG/PNG)', type=['jpg', 'jpeg', 'png'])

if uploaded_file:
    imagem = Image.open(uploaded_file)
    st.image(imagem, caption='🖼️ Imagem enviada', use_column_width=True)

    st.markdown("---")
    st.write('🔎 **Fazendo predição...**')

    # Processar imagem
    img_processada = preprocessar_imagem(imagem)
    predicoes = modelo.predict(img_processada)
    indice_predito = np.argmax(predicoes)
    classe_predita = classes[indice_predito]
    confianca = predicoes[0][indice_predito] * 100

    # Resultado
    st.success(f'✅ **Classe prevista:** {classe_predita}')
    st.info(f'📊 Confiança do modelo: **{confianca:.2f}%**')

    # Mostrar gráfico com todas as classes
    st.markdown("### 🔬 Distribuição das Probabilidades")
    fig, ax = plt.subplots()
    cores = ['brown', 'gold', 'green', 'purple']
    ax.bar(classes, predicoes[0] * 100, color=cores)
    ax.set_ylabel('Confiança (%)')
    ax.set_ylim([0, 100])
    st.pyplot(fig)
