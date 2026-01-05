import streamlit as st
import requests
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List

# ==========================================
# CONFIGURACIÓN DE LA PÁGINA
# ==========================================

st.set_page_config(
    page_title="MoodJournalAI - Emotion Classifier",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# CONSTANTES
# ==========================================

API_URL = "http://localhost:8000/api/predict"
API_URL_ATTENTION = "http://localhost:8000/api/predict/attention"

EMOTION_EMOJIS = {
    "joy": "😊",
    "sadness": "😢",
    "fear": "😨",
    "anger": "😡",
    "love": "💕",
    "surprise": "😮"
}

EMOTION_COLORS = {
    "joy": "#FFD700",      # Dorado
    "sadness": "#4169E1",  # Azul
    "fear": "#8B008B",     # Morado oscuro
    "anger": "#DC143C",    # Rojo
    "love": "#FF69B4",     # Rosa
    "surprise": "#FF8C00"  # Naranja
}

EMOTION_NAMES_ES = {
    "joy": "Alegría",
    "sadness": "Tristeza",
    "fear": "Miedo",
    "anger": "Ira",
    "love": "Amor",
    "surprise": "Sorpresa"
}

# ==========================================
# FUNCIONES
# ==========================================

def predict_emotion(text: str) -> Dict:
    """Llama a la API FastAPI para predecir la emoción"""
    try:
        response = requests.post(
            API_URL,
            json={"text": text},
            timeout=10
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.ConnectionError:
        st.error("❌ No se puede conectar a la API. Asegúrate de que el servidor FastAPI esté corriendo en http://localhost:8000")
        st.info("💡 Ejecuta: `uvicorn backend.api.app.main:app --reload`")
        return None
    except requests.exceptions.Timeout:
        st.error("⏱️ La API tardó demasiado en responder. Intenta de nuevo.")
        return None
    except Exception as e:
        st.error(f"❌ Error al llamar a la API: {str(e)}")
        return None

def predict_emotion_with_attention(text: str) -> Dict:
    """Llama a la API FastAPI para predecir con attention weights"""
    try:
        response = requests.post(
            API_URL_ATTENTION,
            json={"text": text},
            timeout=10
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.ConnectionError:
        st.error("❌ No se puede conectar a la API.")
        return None
    except requests.exceptions.Timeout:
        st.error("⏱️ La API tardó demasiado en responder.")
        return None
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        return None

def create_emotion_bar_chart(all_scores: List[Dict]) -> go.Figure:
    """Crea un gráfico de barras horizontal con los scores de emociones"""
    emotions = [score['emotion'] for score in all_scores]
    scores = [score['score'] for score in all_scores]
    colors = [EMOTION_COLORS[emotion] for emotion in emotions]
    
    fig = go.Figure(data=[
        go.Bar(
            y=emotions,
            x=scores,
            orientation='h',
            marker=dict(
                color=colors,
                line=dict(color='rgba(0,0,0,0.3)', width=1)
            ),
            text=[f"{s:.1%}" for s in scores],
            textposition='outside',
        )
    ])
    
    fig.update_layout(
        title="Distribución de Emociones",
        xaxis_title="Probabilidad",
        yaxis_title="",
        height=400,
        xaxis=dict(range=[0, 1], tickformat='.0%'),
        yaxis=dict(categoryorder='total ascending'),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(size=14)
    )
    
    return fig

def create_emotion_pie_chart(all_scores: List[Dict]) -> go.Figure:
    """Crea un gráfico de pastel con las top 3 emociones"""
    top_3 = all_scores[:3]
    emotions = [score['emotion'] for score in top_3]
    scores = [score['score'] for score in top_3]
    colors = [EMOTION_COLORS[emotion] for emotion in emotions]
    
    fig = go.Figure(data=[
        go.Pie(
            labels=[f"{EMOTION_EMOJIS[e]} {e.capitalize()}" for e in emotions],
            values=scores,
            marker=dict(colors=colors, line=dict(color='white', width=2)),
            textinfo='label+percent',
            textfont=dict(size=16),
            hole=0.3
        )
    ])
    
    fig.update_layout(
        title="Top 3 Emociones",
        height=400,
        showlegend=True,
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig

def create_attention_bar_chart(tokens: List[str], scores: List[float]) -> go.Figure:
    """Crea un gráfico de barras de attention weights"""
    # Combinar tokens y scores, ordenar por score
    combined = list(zip(tokens, scores))
    combined.sort(key=lambda x: x[1], reverse=True)
    
    # Tomar top 10 para que sea legible
    top_n = min(10, len(combined))
    top_tokens, top_scores = zip(*combined[:top_n])
    
    fig = go.Figure(data=[
        go.Bar(
            y=list(top_tokens),
            x=list(top_scores),
            orientation='h',
            marker=dict(
                color=list(top_scores),
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Atención")
            ),
            text=[f"{s:.1%}" for s in top_scores],
            textposition='outside',
        )
    ])
    
    fig.update_layout(
        title="Palabras con Mayor Atención (Top 10)",
        xaxis_title="Nivel de Atención",
        yaxis_title="",
        height=400,
        xaxis=dict(range=[0, 1], tickformat='.0%'),
        yaxis=dict(categoryorder='total ascending'),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(size=12)
    )
    
    return fig

def render_attention_text(tokens: List[str], scores: List[float], emotion: str) -> str:
    """Renderiza texto con palabras resaltadas según attention"""
    emotion_color = EMOTION_COLORS[emotion]
    html = "<div style='font-size: 1.2rem; line-height: 2rem; padding: 1rem; background: #f8f9fa; border-radius: 10px;'>"
    
    for token, score in zip(tokens, scores):
        # Intensidad del color según score
        alpha = score
        background = f"rgba({int(emotion_color[1:3], 16)}, {int(emotion_color[3:5], 16)}, {int(emotion_color[5:7], 16)}, {alpha * 0.6})"
        
        # Tamaño de fuente según importancia
        font_weight = 400 + int(score * 300)  # 400-700
        
        html += f"<span style='background-color: {background}; padding: 2px 4px; margin: 2px; border-radius: 3px; font-weight: {font_weight};'>{token}</span>"
    
    html += "</div>"
    return html


# ==========================================
# INTERFAZ PRINCIPAL
# ==========================================

# Header
st.title("🎭 MoodJournalAI")
st.markdown("### Clasificador de Emociones con IA")
st.markdown("---")

# Sidebar con información
with st.sidebar:
    st.header("ℹ️ Información")
    st.markdown("""
    Esta aplicación utiliza **RoBERTa fine-tuned** para clasificar emociones en texto en inglés.
    
    **Emociones detectadas:**
    - 😊 Joy (Alegría)
    - 😢 Sadness (Tristeza)
    - 😨 Fear (Miedo)
    - 😡 Anger (Ira)
    - 💕 Love (Amor)
    - 😮 Surprise (Sorpresa)
    
    **Tecnologías:**
    - 🤖 Modelo: RoBERTa-base fine-tuned
    - ⚡ API: FastAPI
    - 🎨 Interfaz: Streamlit
    
    **Funcionalidades:**
    - 📊 Predicción de emociones
    - 🔍 Visualización de Attention Weights
    - 📈 Gráficos interactivos
    - 🎯 Explainability (XAI)
    """)
    
    st.markdown("---")
    
    # Estado de la API
    st.subheader("🔌 Estado de la API")
    try:
        health = requests.get("http://localhost:8000/health", timeout=2)
        if health.status_code == 200:
            st.success("✅ API Activa")
        else:
            st.error("❌ API Inactiva")
    except:
        st.error("❌ API No Disponible")

#Input del usuario
st.subheader("📝 Escribe tu texto")
text_input = st.text_area(
    "Ingresa un texto en inglés para analizar su emoción:",
    height=150,
    placeholder="Example: I feel so happy and excited about this amazing project!",
    help="El texto debe estar en inglés. Mínimo 1 carácter, máximo 512."
)

# Toggle para modo attention
col_toggle1, col_toggle2 = st.columns([3, 1])
with col_toggle2:
    show_attention = st.checkbox("🔍 Mostrar Attention", value=False, help="Visualiza qué palabras influyeron más en la predicción")

# Botón de análisis
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    if show_attention:
        analyze_button = st.button("🧠 Analizar con Attention", use_container_width=True, type="primary")
    else:
        analyze_button = st.button("🔍 Analizar Emoción", use_container_width=True, type="primary")

# Ejemplos rápidos
st.markdown("**💡 Ejemplos rápidos:**")
example_col1, example_col2, example_col3 = st.columns(3)

with example_col1:
    if st.button("😊 Ejemplo: Joy"):
        text_input = "I feel so happy and excited today!"
        st.rerun()

with example_col2:
    if st.button("😢 Ejemplo: Sadness"):
        text_input = "I am feeling really sad and lonely right now"
        st.rerun()

with example_col3:
    if st.button("😨 Ejemplo: Fear"):
        text_input = "I am terrified about what might happen tomorrow"
        st.rerun()

st.markdown("---")

# Procesamiento y resultados
if analyze_button and text_input.strip():
    with st.spinner("🤔 Analizando emoción..."):
        if show_attention:
            result = predict_emotion_with_attention(text_input)
        else:
            result = predict_emotion(text_input)
    
    if result:
        # Resultado principal
        st.success("✅ Análisis completado")
        
        emotion = result['predicted_emotion']
        confidence = result['confidence']
        emoji = EMOTION_EMOJIS[emotion]
        emotion_name_es = EMOTION_NAMES_ES[emotion]
        
        # Mostrar emoción principal con estilo
        st.markdown(f"""
        <div style="text-align: center; padding: 2rem; background: linear-gradient(135deg, {EMOTION_COLORS[emotion]}22 0%, {EMOTION_COLORS[emotion]}44 100%); border-radius: 15px; margin: 1rem 0;">
            <h1 style="font-size: 5rem; margin: 0;">{emoji}</h1>
            <h2 style="margin: 0.5rem 0; color: {EMOTION_COLORS[emotion]};">{emotion.upper()}</h2>
            <p style="font-size: 1.2rem; color: #666; margin: 0;">({emotion_name_es})</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Métrica de confianza
        st.metric(
            label="Confianza de la predicción",
            value=f"{confidence:.1%}",
            delta=None
        )
        
        # Si hay datos de attention, mostrarlos primero
        if show_attention and 'attention' in result:
            st.markdown("### 🔍 Visualización de Attention")
            st.markdown("**Palabras resaltadas según su importancia en la predicción:**")
            st.info("💡 Las palabras más oscuras/gruesas tuvieron mayor influencia en la predicción.")
            
            attention_data = result['attention']
            tokens = attention_data['tokens']
            scores = attention_data['scores']
            
            # Texto resaltado
            highlighted_html = render_attention_text(tokens, scores, emotion)
            st.markdown(highlighted_html, unsafe_allow_html=True)
            
            st.markdown("---")
            
            # Gráficos de attention
            att_col1, att_col2 = st.columns(2)
            
            with att_col1:
                # Gráfico de barras de attention
                attention_chart = create_attention_bar_chart(tokens, scores)
                st.plotly_chart(attention_chart, use_container_width=True)
            
            with att_col2:
                # Top 5 palabras más importantes
                st.markdown("#### 📌 Top 5 Palabras Clave")
                combined = list(zip(tokens, scores))
                combined.sort(key=lambda x: x[1], reverse=True)
                
                for i, (token, score) in enumerate(combined[:5], 1):
                    st.markdown(f"**{i}.** `{token}` - {score:.1%}")
            
            st.markdown("---")
        
        # Gráficos de emociones
        st.markdown("### 📊 Distribución de Emociones")
        
        chart_col1, chart_col2 = st.columns(2)
        
        with chart_col1:
            # Gráfico de barras
            bar_chart = create_emotion_bar_chart(result['all_scores'])
            st.plotly_chart(bar_chart, use_container_width=True)
        
        with chart_col2:
            # Gráfico de pastel (top 3)
            pie_chart = create_emotion_pie_chart(result['all_scores'])
            st.plotly_chart(pie_chart, use_container_width=True)
        
        # Tabla de detalles
        with st.expander("📋 Ver detalles completos"):
            st.json(result)

elif analyze_button:
    st.warning("⚠️ Por favor, ingresa un texto para analizar.")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 1rem;">
    Desarrollado con ❤️ usando RoBERTa Fine-tuning | MoodJournalAI v1.0
</div>
""", unsafe_allow_html=True)
