import cv2
import av
import numpy as np
import mediapipe as mp
import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration

# --- Configuração da Página ---
st.set_page_config(page_title="Mestre Mobile AI", layout="centered")

st.title("🖐️ Detector de Mãos - Elite Mobile")
st.markdown("### Processamento em Python 3.10 | Esqueleto Rosa")

# --- Configuração do MediaPipe (Rosa Choque) ---
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Definição das Cores e Estilo
# Cor: (R:255, G:0, B:255) -> Magenta
PINK_COLOR = (255, 0, 255)
landmark_style = mp_drawing.DrawingSpec(color=PINK_COLOR, thickness=3, circle_radius=3)
connection_style = mp_drawing.DrawingSpec(color=PINK_COLOR, thickness=3, circle_radius=2)

# --- Classe Processadora de Vídeo (O Coração do WebRTC) ---
class HandDetectorProcessor:
    def __init__(self):
        # Inicializa o MediaPipe apenas uma vez para performance
        self.hands = mp_hands.Hands(
            model_complexity=0,  # 0 = Leve (Rápido para Celular)
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            max_num_hands=2
        )

    def recv(self, frame):
        # 1. Converte frame WebRTC (av) para NumPy (OpenCV)
        img = frame.to_ndarray(format="bgr24")

        # 2. Espelha a imagem (opcional, fica mais natural tipo espelho)
        img = cv2.flip(img, 1)

        # 3. Processamento MediaPipe
        # Converte para RGB pois o MP não lê BGR
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.hands.process(img_rgb)

        # 4. Desenha o esqueleto se achar mãos
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    img,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    landmark_drawing_spec=landmark_style,
                    connection_drawing_spec=connection_style
                )

        # 5. Retorna o frame processado de volta para a tela
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# --- Configuração de Rede (Essencial para Mobile 4G/5G) ---
# Sem isso, o vídeo trava fora do Wi-Fi
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# --- Componente Visual na Tela ---
st.info("Clique em 'START' e dê permissão para a câmera.")

webrtc_streamer(
    key="hand-detection",
    mode=WebRtcMode.SENDRECV,       # Envia vídeo e recebe vídeo processado
    rtc_configuration=RTC_CONFIGURATION,
    video_processor_factory=HandDetectorProcessor,
    media_stream_constraints={
        "video": {"facingMode": "environment"}, # Tenta pegar câmera traseira
        "audio": False
    },
    async_processing=True,
)

st.success("Sistema Operacional: Python 3.10 | Status: Online")
