import streamlit as st
from PIL import Image
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from twilio.rest import Client
import requests


# Function to preprocess image for CNN model
def preprocess_image(img):
    img = img.resize((256, 256))  # Resize image to match model's expected sizing
    img = np.array(img)  
    img = img / 255.0  # Normalize pixel values to between 0 and 1
    img = np.expand_dims(img, axis=0)  # Expand dimensions to match model's expected input shape
    return img

# Function to load CNN model
def load_cnn_model(model_path):
    if 'cnn_model' not in st.session_state:
        st.session_state.cnn_model = load_model(model_path, compile=False)  # Load model without compilation
    return st.session_state.cnn_model

# Function to predict using CNN model
def predict_cnn(image, model):
    processed_img = preprocess_image(image)
    prediction = model.predict(processed_img)
    return prediction

# Function to display CNN model prediction result
def display_cnn_result(prediction):
    class_names = ["smoke", "fire", "non fire"]  # Adjust based on your model's output classes
    st.subheader('CNN Model Prediction:')
    st.write(f'Class: {class_names[np.argmax(prediction)]}')
    st.write(f'Confidence: {prediction[0][np.argmax(prediction)] * 100:.2f}%')

# Function to perform real-time detection using CNN model and webcam
def process_webcam_cnn(model):
    class_names = ["smoke", "fire", "non fire"]
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("Error: Could not open webcam.")
        return

    frame_window = st.image([])  # Create a placeholder for the video frames
    stop_button = st.button('Stop CNN Real-time Detection')  # Create a button to stop the detection

    while cap.isOpened() and not stop_button:
        ret, frame = cap.read()
        if not ret:
            st.error("Error: Failed to capture image.")
            break

        input_frame = Image.fromarray(frame)
        prediction = predict_cnn(input_frame, model)
        class_index = np.argmax(prediction)
        class_name = class_names[class_index]
        probability = prediction[0][class_index]

        if class_name != "non fire" and probability > 0.5:
            text = f'Class: {class_name}, Probability: {probability:.2f}'
            cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            # Send SMS with detection details and location
            send_sms(f'Fire or smoke detected with probability {probability:.2f}!', get_location())

        frame_window.image(frame, channels="BGR")  # Update the frame in the placeholder

    cap.release()

# Function to preprocess image for MobileNetV2 model
def preprocess_mobilenet_image(img):
    img = img.resize((224, 224))  # Resize image to match model's expected sizing
    img = np.array(img)  # Convert to numpy array
    img = np.expand_dims(img, axis=0)  # Expand dimensions to match model's expected input shape
    return img

# Function to load MobileNetV2 model
def load_mobilenet_model(model_path):
    model = load_model(model_path)
    return model

# Function to predict using MobileNetV2 model
def predict_mobilenet(image, model):
    processed_img = preprocess_mobilenet_image(image)
    prediction = model.predict(processed_img)
    return prediction

# Function to display MobileNetV2 model prediction result
def display_mobilenet_result(prediction):
    class_names = ["fire", "smoke", "non fire"]  
    st.subheader('MobileNetV2 Model Prediction:')
    st.write(f'Class: {class_names[np.argmax(prediction)]}')
    st.write(f'Confidence: {prediction[0][np.argmax(prediction)] * 100:.2f}%')

# Function to perform real-time detection using MobileNetV2 model and webcam
def process_webcam_mobilenet(model):
    class_names = ["smoke", "fire", "non fire"]
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("Error: Could not open webcam.")
        return

    frame_window = st.image([])  # Create a placeholder for the video frames
    stop_button = st.button('Stop MobileNetV2 Real-time Detection')  # Create a button to stop the detection

    while cap.isOpened() and not stop_button:
        ret, frame = cap.read()
        if not ret:
            st.error("Error: Failed to capture image.")
            break

        input_frame = Image.fromarray(frame)
        prediction = predict_mobilenet(input_frame, model)
        class_index = np.argmax(prediction)
        class_name = class_names[class_index]
        probability = prediction[0][class_index]

        if class_name != "non fire" and probability > 0.5:
            text = f'Class: {class_name}, Probability: {probability:.2f}'
            cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            # Send SMS with detection details and location
            send_sms(f'Fire or smoke detected with probability {probability:.2f}!', get_location())

        frame_window.image(frame, channels="BGR")  # Update the frame in the placeholder

    cap.release()

# Function to get location using IPStack API
def get_location():
    ipstack_access_key = '4f2395559a1cf6cb8d7abaff61da62ee'
    url = f'http://api.ipstack.com/check?access_key={ipstack_access_key}'

    try:
        response = requests.get(url)
        data = response.json()
        lat = data['latitude']
        lng = data['longitude']
        return lat, lng
    except Exception as e:
        st.error(f"Error fetching location: {str(e)}")
        return None

# Function to send SMS using Twilio API
def send_sms(message, location=None):
    account_sid = 'ACf1eb47d607fc4125f8d34a797a5922f1'  
    auth_token = '561bc564dae5bbb0fd2d0bf8824cb481'  
    twilio_phone_number = '+13602051482'  
    to_phone_number = '+917004001927' 

    if location:
        location_url = f'https://www.google.com/maps/search/?api=1&query=12.937827506064064,77.69253813505036'
        message += f'\nLocation: {location_url}'

    client = Client(account_sid, auth_token)
    client.messages.create(
        body=message,
        from_=twilio_phone_number,
        to=to_phone_number
    )

# Main function for Streamlit app
def main():
    st.title('Fire and Smoke Detection App')

    # Sidebar navigation for selecting model
    st.sidebar.title('Select Model')
    model_choice = st.sidebar.radio('Choose Model:', ('CNN (Uploaded Image)', 'CNN (Real-time Detection)', 'MobileNetV2 (Uploaded Image)', 'MobileNetV2 (Real-time Detection)'))

    cnn_model_path = 'C:\\Users\\Madhura M R\\OneDrive\\Desktop\\New folder (3)\\fire and smoke\\fire and smoke\\new_model.h5'  
    if 'cnn_model' not in st.session_state:
        st.session_state.cnn_model = load_cnn_model(cnn_model_path)
    cnn_model = st.session_state.cnn_model

    # Load MobileNetV2 model
    mobilenet_model_path = 'C:\\Users\\Madhura M R\\OneDrive\\Desktop\\New folder (3)\\fire and smoke\\fire and smoke\\mobileNetv2-model.h5' 
    if 'mobile_model' not in st.session_state:
        st.session_state.mobile_model = load_mobilenet_model(mobilenet_model_path)
    mobile_model = st.session_state.mobile_model  # Assign loaded model to mobile_model variable

    # Upload image for prediction using CNN model
    if model_choice == 'CNN (Uploaded Image)':
        st.sidebar.title('Upload Image (CNN)')
        uploaded_file = st.sidebar.file_uploader('Choose an image...', type=['jpg', 'jpeg', 'png'])

        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption='Uploaded Image', use_column_width=True)

            # Button to trigger prediction
            if st.sidebar.button('Predict (CNN)', key='predict_button'):
                prediction = predict_cnn(image, cnn_model)
                display_cnn_result(prediction)

    # Real-time detection using CNN model
    elif model_choice == 'CNN (Real-time Detection)':
        st.header('Real-time Fire and Smoke Detection using CNN')
        st.sidebar.warning('CNN model requires webcam.')

        if st.sidebar.button('Start CNN Real-time Detection', key='cnn_realtime_button'):
            process_webcam_cnn(cnn_model)

    # Upload image for prediction using MobileNetV2 model
    elif model_choice == 'MobileNetV2 (Uploaded Image)':
        st.sidebar.title('Upload Image (MobileNetV2)')
        uploaded_file = st.sidebar.file_uploader('Choose an image...', type=['jpg', 'jpeg', 'png'])

        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption='Uploaded Image', use_column_width=True)

            # Button to trigger prediction
            if st.sidebar.button('Predict (MobileNetV2)', key='mobilenetv2_predict_button'):
                prediction = predict_mobilenet(image, mobile_model)
                display_mobilenet_result(prediction)

    # Real-time detection using MobileNetV2 model
    elif model_choice == 'MobileNetV2 (Real-time Detection)':
        st.header('Real-time Fire and Smoke Detection using MobileNetV2')
        st.sidebar.warning('MobileNetV2 model requires webcam.')

        if st.sidebar.button('Start MobileNetV2 Real-time Detection', key='mobilenetv2_realtime_button'):
            process_webcam_mobilenet(mobile_model)

if __name__ == "__main__":
    main()
