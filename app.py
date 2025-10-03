import cv2
import numpy as np
import streamlit as st
from keras.models import load_model
from keras.preprocessing.image import img_to_array

try:
    emotion_model = load_model('emotion_detector.h5')
except Exception as e:
    st.error(f"Error loading the model: {e}")

'''
First, you need to scan the webcam image and locate where the face is. This is the job of the Haar Cascade classifier you've loaded. It's an efficient algorithm that is good at identifying objects, in this case, faces.
'''
try:
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
except Exception as e:
    st.error(f"Error loading the Haar Cascade model: {e}")

EMOTIONS = ["Angry","Disgust","Fear","Happy","Neutral","Sad","Surprise"]

def detect_emotion(frame):
    if frame is None:
        return None
    
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=5,minSize=(30,30))

    for (x,y,w,h) in faces:
        roi_gray = gray[y:y+h,x:x+w]
        roi_gray = cv2.resize(roi_gray,(48,48),interpolation=cv2.INTER_AREA)

        if np.sum([roi_gray]) != 0:
            roi = roi_gray.astype('float')/255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi,axis=0)

            preds = emotion_model.predict(roi)[0]
            label = EMOTIONS[preds.argmax()]
            probability = preds[preds.argmax()]

            label_position = (x,y-10)
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
            cv2.putText(frame,f"{label}:{probability:.2f}",label_position,cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),2)
        else:
            cv2.putText(frame,'No Face Found',(20,60),cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),2)
    return frame

def main():
    st.set_page_config(page_title="Real-Time Emotion Detection", page_icon="😊")
    st.title("Real-Time Face Emotion Detection")
    st.write("This application uses your webcam to detect faces and predict their emotions in real-time.")
    st.write("Make sure to allow webcam access when prompted by your browser.")

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        st.error("Cannot open Webcam")
        return
    
    frame_placeholder = st.empty()

    stop_button_pressed = st.button("Stop")

    while cap.isOpened() and not stop_button_pressed:
        ret,frame = cap.read()
        if not ret:
            st.write("The video capture has ended.")
            break

        processed_frame = detect_emotion(frame)

        if processed_frame is not None:
            frame_placeholder.image(cv2.cvtColor(processed_frame,cv2.COLOR_BGR2RGB), channels="RGB")
        
        if cv2.waitKey(1) & 0xFF == ord('q') or stop_button_pressed:
            break
    
    cap.release()
    cv2.destroyAllWindows()
    st.write("Webcam Stopped..")

if __name__ == '__main__':
    main()