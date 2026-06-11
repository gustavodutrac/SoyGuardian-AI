import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# =====================================
# CONFIGURAÇÃO DA PÁGINA
# =====================================

st.set_page_config(
    page_title="SoyGuardian AI",
    page_icon="🌱",
    layout="centered"
)

# =====================================
# CARREGAR MODELO
# =====================================

@st.cache_resource
def carregar_modelo():
    return tf.keras.models.load_model(
        "models/soybean_model.keras"
    )

model = carregar_modelo()

# =====================================
# CLASSES
# =====================================

classes = [
    "Lagarta",
    "Diabrotica speciosa",
    "Saudável"
]

# =====================================
# CABEÇALHO
# =====================================

st.title("🌱 SoyGuardian AI")

st.subheader(
    "Sistema Inteligente de Monitoramento Precoce de Estresse Foliar em Soja"
)

st.success(
    "IA operacional e pronta para análise."
)

st.write(
    "Envie uma imagem de uma folha de soja para análise automática."
)

# =====================================
# UPLOAD
# =====================================

uploaded_file = st.file_uploader(
    "Escolha uma imagem",
    type=["jpg", "jpeg", "png"]
)

# =====================================
# PROCESSAMENTO
# =====================================

if uploaded_file is not None:

    try:

        # Converter imagem para RGB
        image = Image.open(
            uploaded_file
        ).convert("RGB")

        st.image(
            image,
            caption="Imagem enviada",
            width="stretch"
        )

        # Redimensionar
        image_resized = image.resize(
            (224, 224)
        )

        # Converter para array
        image_array = np.array(
            image_resized
        )

        # Adicionar dimensão batch
        image_array = np.expand_dims(
            image_array,
            axis=0
        )

        # =====================================
        # PREDIÇÃO
        # =====================================

        prediction = model.predict(
            image_array,
            verbose=0
        )

        probabilidades = {
            "Lagarta": float(prediction[0][0]),
            "Diabrotica": float(prediction[0][1]),
            "Saudável": float(prediction[0][2])
        }

        st.write(
            "### Probabilidades por classe"
        )

        st.write(probabilidades)

        st.bar_chart(probabilidades)

        class_index = np.argmax(
            prediction
        )

        confidence = float(
            np.max(prediction) * 100
        )

        diagnostico = classes[
            class_index
        ]

        # =====================================
        # NÍVEL DE RISCO
        # =====================================

        if diagnostico == "Saudável":
            risco = "🟢 Baixo"

        elif diagnostico == "Lagarta":
            risco = "🟡 Médio"

        else:
            risco = "🔴 Alto"

        # =====================================
        # RESULTADOS
        # =====================================

        st.success(
            f"Diagnóstico: {diagnostico}"
        )

        st.info(
            f"Confiança da IA: {confidence:.2f}%"
        )

        st.info(
            f"Nível de risco agronômico: {risco}"
        )

        st.progress(
            min(
                int(confidence),
                100
            )
        )

        # =====================================
        # BAIXA CONFIANÇA
        # =====================================

        if confidence < 70:

            st.warning(
                "A IA não está totalmente confiante. Recomenda-se utilizar outra imagem para confirmação."
            )

        # =====================================
        # RECOMENDAÇÕES
        # =====================================

        if diagnostico == "Saudável":

            st.success(
                "Planta saudável."
            )

            st.info("""
Recomendações:

• Continuar monitoramento semanal.

• Manter manejo nutricional adequado.

• Não há necessidade de intervenção imediata.
""")

        elif diagnostico == "Lagarta":

            st.warning(
                "Possível presença de lagartas."
            )

            st.info("""
Recomendações:

• Realizar inspeção visual da área.

• Verificar percentual de desfolha.

• Avaliar necessidade de controle biológico.

• Monitorar evolução nas próximas 48 horas.
""")

        elif diagnostico == "Diabrotica speciosa":

            st.error(
                "Possível infestação de Diabrotica speciosa."
            )

            st.info("""
Recomendações:

• Inspecionar áreas vizinhas.

• Avaliar danos foliares.

• Monitorar expansão do foco.

• Consultar engenheiro agrônomo responsável.
""")

    except Exception as erro:

        st.error(
            f"Erro ao processar imagem: {erro}"
        )

# =====================================
# EXPLICAÇÃO DA IA
# =====================================

with st.expander("Como a IA funciona?"):

    st.write("""
O SoyGuardian AI utiliza Deep Learning e Visão Computacional
para analisar imagens de folhas de soja.

O modelo foi treinado utilizando a arquitetura EfficientNetB0,
capaz de identificar padrões visuais associados a:

• Lagartas

• Diabrotica speciosa

• Plantas saudávei

O objetivo é auxiliar produtores rurais na identificação
precoce de problemas fitossanitários e apoiar a tomada
de decisão no campo.
""")

# =====================================
# RODAPÉ
# =====================================

st.markdown("---")

st.caption(
    "Projeto Acadêmico de Deep Learning | SoyGuardian AI © 2026"
)