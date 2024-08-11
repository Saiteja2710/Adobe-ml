import numpy as np
import cv2
import streamlit as st
from PIL import Image

def detect_symmetry(image):
    # Convert RGB to BGR (for OpenCV processing)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    all_contour_points = []

    for contour in contours:
        all_contour_points.extend(contour.reshape(-1, 2))

    if len(all_contour_points) == 0:
        return None

    all_contour_points = np.array(all_contour_points)
    hull = cv2.convexHull(all_contour_points)

    moments = cv2.moments(hull)
    if moments['m00'] != 0:
        cX = int(moments['m10'] / moments['m00'])
        cY = int(moments['m01'] / moments['m00'])

        cv2.circle(image, (cX, cY), 7, (255, 0, 0), -1)
        cv2.putText(image, f'Centroid ({cX}, {cY})', (cX - 20, cY - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
        best_line = None
        min_diff = float('inf')

        for angle in np.linspace(0, 180, 360):
            a = np.cos(np.deg2rad(angle))
            b = np.sin(np.deg2rad(angle))
            d = -(a * cX + b * cY)
            left_count = 0
            right_count = 0

            for point in all_contour_points:
                if a * point[0] + b * point[1] + d < 0:
                    left_count += 1
                else:
                    right_count += 1

            diff = abs(left_count - right_count)
            if diff < min_diff:
                min_diff = diff
                best_line = (a, b, d)

        if best_line:
            a, b, d = best_line
            x0 = int(-d * a - 1000 * b)
            y0 = int(-d * b + 1000 * a)
            x1 = int(-d * a + 1000 * b)
            y1 = int(-d * b - 1000 * a)
            cv2.line(image, (x0, y0), (x1, y1), (0, 255, 255), 2)
    
    # Convert BGR back to RGB (for display in Streamlit)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

# Streamlit UI
st.title('Symmetry Detector')

# Handle image upload
uploaded_image = st.file_uploader("Choose an image file (png, jpg)", type=["png", "jpg", "jpeg"])

if uploaded_image is not None:
    try:
        image = Image.open(uploaded_image)
        image = np.array(image)

        processed_image = detect_symmetry(image)

        if processed_image is not None:
            processed_image = Image.fromarray(processed_image)
            # Resize the image
            resized_image = processed_image.resize((processed_image.width // 2, processed_image.height // 2))
            st.image(resized_image, caption='Processed Image with Symmetry Detection', use_column_width=True)
        else:
            st.error("Symmetry could not be detected. Please try another image.")
    except Exception as e:
        st.error(f"An error occurred: {e}")
else:
    st.info("Please upload an image file to proceed.")
