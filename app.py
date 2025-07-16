import streamlit as st
import numpy as np
import time
import tensorflow as tf
from PIL import Image
import cv2
import tempfile
import os
import librosa
import tensorflow_hub as hub



st.set_page_config(page_title="AI Deepfake Detector", page_icon="🤖", layout="wide")

#  Load the pre-trained deepfake detection model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("deepfake_detection_model.h5")  

model = load_model()

#  Function to process images correctly
def predict_image(image):
    image = image.resize((96, 96))  # Adjusted to match model input size
    image_array = np.array(image) / 255.0  # Normalize
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension

    # Get prediction from model
    prediction = model.predict(image_array)[0][0]  # Adjust index if needed
    confidence = prediction * 100
    is_fake = confidence > 50  # Assuming 50%+ confidence means fake

    return is_fake, confidence


def load_model1():
    return tf.keras.models.load_model("audio_deepfake_model.h5") 

model1=load_model1()

def predict_audio(audio_file):
    try:
        temp_audio_path = tempfile.NamedTemporaryFile(delete=False, suffix='.wav').name
        with open(temp_audio_path, 'wb') as f:
            f.write(audio_file.read())

        y, sr = librosa.load(temp_audio_path, sr=16000)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        mfccs_processed = np.mean(mfccs.T, axis=0)

        input_audio = np.expand_dims(mfccs_processed, axis=0)

        prediction = model1.predict(input_audio)[0]
        real_conf, fake_conf = prediction[0], prediction[1]

        st.write(f"🔍 Model Output: Real: {real_conf:.2f}, Fake: {fake_conf:.2f}")

        is_fake = fake_conf > real_conf
        confidence = fake_conf * 100 if is_fake else real_conf * 100

        return is_fake, confidence

    except Exception as e:
        st.error(f"Error processing audio: {str(e)}")
        return None, None

#  Function to process videos
def predict_video(video_file):
    # Create a temporary file
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    temp_file_path = temp_file.name
    temp_file.write(video_file.read())  
    temp_file.close()  # Explicitly close the file

    # Open video with OpenCV
    cap = cv2.VideoCapture(temp_file_path)
    frame_count = 0
    fake_frames = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert frame to PIL image and predict
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame)
        is_fake, _ = predict_image(image)
        
        frame_count += 1
        if is_fake:
            fake_frames += 1

    cap.release()  # Ensure video capture is closed
    cv2.destroyAllWindows()  # Destroy any OpenCV windows

    # Ensure the file is not in use before deleting
    try:
        os.remove(temp_file_path)
    except PermissionError:
        print(f"Warning: Unable to delete temporary file {temp_file_path}")

    confidence = (fake_frames / frame_count) * 100 if frame_count > 0 else 0
    is_fake = confidence > 50  # Assuming 50%+ frames being fake means video is fake

    return is_fake, confidence


# Sci-Fi Themed CSS + Correct Background Image
st.markdown(f"""
    <style>
        body {{
            background-color: #0D0D0D; 
            color: white; 
            font-family: Arial, sans-serif; 
        }}
        .stApp {{
            background-image: url('https://png.pngtree.com/thumb_back/fh260/back_our/20190620/ourmid/pngtree-black-cool-sci-fi-background-promotion-main-map-image_149429.jpg'); 
            background-size: cover; 
            background-attachment: fixed;
        }}
        .title {{
            font-size: 48px; 
            font-weight: bold; 
            color: white; 
            text-align: center; 
            text-shadow: 0px 0px 10px cyan;
        }}
        .description {{
            font-size: 20px; 
            text-align: center; 
            color: lightgray; 
            margin-bottom: 20px;
            padding: 10px;
            background: rgba(0, 0, 0, 0.6); 
            border-radius: 10px;
        }}
        .stFileUploader {{
            color: white;
            border: 2px solid cyan !important; 
            border-radius: 10px; 
            padding: 10px;
        }}
        .stButton>button {{
            background-color: cyan !important; 
            color: black !important; 
            font-size: 18px; 
            border-radius: 10px; 
            padding: 10px;
        }}
        .result-box {{
            background: rgba(0, 255, 255, 0.3); 
            padding: 20px; 
            border-radius: 10px; 
            text-align: center; 
            font-size: 22px; 
            font-weight: bold; 
            box-shadow: 0px 0px 10px cyan;
        }}
    </style>
""", unsafe_allow_html=True)

#  Page Header with Enhanced Text
st.markdown("<h2 class='title'>🤖 AI Deepfake Detector</h2>", unsafe_allow_html=True)
st.markdown("""
    <p class='description'>
        Deepfakes are AI-generated images ,audios and videos designed to mimic real people with incredible accuracy. 
        While some deepfakes are harmless, others can be used for misinformation or fraud. <br><br>
        This AI-powered tool helps detect whether an image or video is real or artificially generated.
        Upload a file, and our model will analyze it with state-of-the-art deep learning algorithms!
    </p>
""", unsafe_allow_html=True)

#  Upload Section
file = st.file_uploader("Upload an Image, Video or Audio", type=["jpg", "jpeg", "png", "mp4", "wav","mp3"], help="Supports JPG, PNG, MP4,MP3, WAV (200MB max)")
#  Prediction Logic
if file:
    file_type = file.name.split(".")[-1].lower()
    if file_type in ["jpg", "jpeg", "png"]:
        image = Image.open(file)
        st.image(image, caption="Uploaded Image", use_container_width=True)
        
        with st.spinner("🔍 Analyzing image..."):
            is_fake, confidence = predict_image(image)

        #  Display Result
        st.markdown(f"<div class='result-box'>🚨 { 'Fake' if is_fake else 'Real' } | Confidence: {confidence:.2f}%</div>", unsafe_allow_html=True)

    elif file_type == "mp4":
        st.video(file)

        with st.spinner("🔍 Analyzing video... This may take a while."):
            is_fake, confidence = predict_video(file)

        #  Display Result
        st.markdown(f"<div class='result-box'>🚨 { 'Fake' if is_fake else 'Real' } | Confidence: {confidence:.2f}%</div>", unsafe_allow_html=True)
    
    elif file_type in [ "wav","mp3","mp4"]:
        st.audio(file)

        with st.spinner("🔍 Analyzing audio..."):
            is_fake, confidence = predict_audio(file)

        if confidence is not None:
            st.markdown(f"<div class='result-box'>🚨 { 'Fake' if is_fake else 'Real' } | Confidence: {confidence:.2f}%</div>", unsafe_allow_html=True)



#  Footer
st.markdown("<p style='text-align:center; font-size:14px; color:gray;'>Powered by AI | Developed by Aaradhya & Alisha</p>", unsafe_allow_html=True)