import streamlit as st
import cv2
import numpy as np
import os
import uuid
# *** OpenCV Dehazing Functions ***

def estimate_atmospheric_light(img, mean_of_top_percentile=0.1):
    # Find the number of pixels to take from the top percentile
    num_pixels = int(np.prod(img.shape[:2]) * mean_of_top_percentile)

    # Find the maximum pixel value for each channel
    max_channel_vals = np.max(np.max(img, axis=0), axis=0)

    # Sort the channel values in descending order
    sorted_vals = np.argsort(max_channel_vals)[::-1]

    # Take the highest pixel values from each channel
    atmospheric_light = np.zeros((1, 1, 3), np.uint8)
    for channel in range(3):
        atmospheric_light[0, 0, channel] = np.sort(img[:, :, sorted_vals[channel]].ravel())[-num_pixels]

    return atmospheric_light

def fast_visibility_restoration(frame, atmospheric_light, tmin=0.1, A=1.0, omega=0.95, guided_filter_radius=40, gamma=0.7):
    # Normalize the frame and atmospheric light
    normalized_frame = frame.astype(np.float32) / 255.0
    normalized_atmospheric_light = atmospheric_light.astype(np.float32) / 255.0

    # Compute the transmission map
    transmission_map = 1 - omega * cv2.cvtColor(normalized_frame, cv2.COLOR_BGR2GRAY) / cv2.cvtColor(normalized_atmospheric_light, cv2.COLOR_BGR2GRAY)

    # Apply the soft matting guided filter to the transmission map
    guided_filter = cv2.ximgproc.createGuidedFilter(normalized_frame, guided_filter_radius, eps=1.0)
    transmission_map = guided_filter.filter(transmission_map)

    # Apply the gamma correction to the transmission map
    transmission_map = np.power(transmission_map, gamma)

    # Threshold the transmission map to ensure a minimum value
    transmission_map = np.maximum(transmission_map, tmin)

    # Compute the dehazed image
    dehazed_frame = (normalized_frame - normalized_atmospheric_light) / np.expand_dims(transmission_map, axis=2) + normalized_atmospheric_light

    # Apply the A parameter to the dehazed image
    dehazed_frame = A * dehazed_frame

    # Normalize the dehazed image and convert to 8-bit color
    dehazed_frame = np.uint8(np.clip(dehazed_frame * 255.0, 0, 255))

    return dehazed_frame

def process_video(input_video_path, output_video_path):
    """Processes a video file, applying dehazing, and saves the output."""

    video = cv2.VideoCapture(input_video_path)
    ret, frame = video.read()
    atmospheric_light = estimate_atmospheric_light(frame)

    frame_width, frame_height = frame.shape[1], frame.shape[0]
    out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), 20.0, (frame_width, frame_height))

    while True:
        ret, frame = video.read()
        if not ret:
            break

        dehazed_frame = fast_visibility_restoration(frame, atmospheric_light)
        out.write(dehazed_frame)

    video.release()
    out.release()

# *** Streamlit App Code ***
st.set_page_config(page_title="Dehazing using Computer Vision and Image Processing", layout="centered")

st.markdown("<h1 style='text-align: center;'>Dehazing using Computer Vision and Image Processing</h1>", unsafe_allow_html=True)
st.subheader("Enhance the clarity and visual appeal of your outdoor videos!")

st.markdown("<text style='text-align: justify; font-size: 25px'>Enhance the clarity and visual appeal of your outdoor videos. Our dehazing tool improves visibility, contrast, and color fidelity, revealing the true details hidden by atmospheric conditions</text>", unsafe_allow_html=True)

st.markdown("""
<h1 style="font-size: 25px;">How to Upload Your Video</h1> 
<p style="font-size: 25px;">It's super easy to get your hazy video fixed! Here's what you do:</p>
""", unsafe_allow_html=True)

st.subheader("1. Find the 'Upload' Button")
st.write("Look for a big button that says 'Choose a video file', 'Browse', or something similar. This button is usually in a section titled 'Upload Your Video'.")

st.subheader("2. Open the File Explorer")
st.write("When you click the button, a window will pop up. This window is like a map of all the files on your computer.  Navigate through the folders until you find the video you want to dehaze.")

st.subheader("3. Select Your Video")
st.write("Once you find your video file, double-click on its name. This tells the website which video you want to work on.")

st.subheader("4. A Moment of Patience")
st.write("Uploading your video takes a little time, especially if it's a big file. You might see a progress bar or something that spins.  Just hang tight!")

st.subheader("5. Witness the Transformation")
st.write("The website will use its special technology to clear the haze from your video. When it's done, the new and improved version will appear!")


# Video Upload and Processing 
st.header("Upload Your Video")
uploaded_video = st.file_uploader("Choose a video file", type=['mp4'])

if uploaded_video:
    # Create 'Output' folder if it doesn't exist
    output_dir = "Output"
    os.makedirs(output_dir, exist_ok=True)

    temp_file_location = "temp_video.mp4"
    unique_id = str(uuid.uuid4())  # Generate a unique ID
    output_file_location = os.path.join(output_dir, f"nostreamlit_{unique_id}.mp4")
    conversion_output_file = os.path.join(output_dir, f"yesstreamlit_{unique_id}.mp4")

    with open(temp_file_location, "wb") as f:
        f.write(uploaded_video.getbuffer())

    # Load the uploaded video for display
    uploaded_video_bytes = uploaded_video.getvalue()

    process_video(temp_file_location, output_file_location)

    # ffmpeg Conversion with overwrite
    ffmpeg_command = f"ffmpeg.exe -y -i {output_file_location} -vcodec libx264 {conversion_output_file}"
    os.system(ffmpeg_command)

    # Display Processed Video
    with open(conversion_output_file, 'rb') as f:
        processed_video_bytes = f.read()

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original")
        st.video(uploaded_video_bytes)
    with col2:
        st.subheader("Dehazed")
        st.video(processed_video_bytes)

    # Download Button
    st.download_button(label="Download Dehazed Video", data=processed_video_bytes, file_name=conversion_output_file)

# Sample Videos
video_file_before = open(r"C:\Users\palam\OneDrive\Desktop\EXP\Video_fog_removal-main\Sample_resources\base_I.mp4", 'rb')
video_bytes_before = video_file_before.read()

video_file_after = open(r"C:\Users\palam\OneDrive\Desktop\EXP\Video_fog_removal-main\Sample_resources\opc.mp4", 'rb')
video_bytes_after = video_file_after.read()

col1, col2 = st.columns(2)
with col1:
    st.subheader("Before")
    st.video(video_bytes_before)
with col2:
    st.subheader("After")
    st.video(video_bytes_after)