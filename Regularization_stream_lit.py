import numpy as np
import matplotlib.pyplot as plt
import cv2
import streamlit as st
import tempfile
import os

# Step 1: Read CSV and Plot
def read_csv(csv_path):
    np_path_XYs = np.genfromtxt(csv_path, delimiter=',')
    path_XYs = []

    for i in np.unique(np_path_XYs[:, 0]):
        npXYs = np_path_XYs[np_path_XYs[:, 0] == i][:, 1:]
        XYs = []
        for j in np.unique(npXYs[:, 0]):
            XY = npXYs[npXYs[:, 0] == j][:, 1:]
            XYs.append(XY)
        path_XYs.append(XYs)
    
    return path_XYs

def plot(paths_XYs, output_image_path):
    fig, ax = plt.subplots(tight_layout=True, figsize=(4, 4))  # Smaller figure size
    colours = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    
    for i, XYs in enumerate(paths_XYs):
        c = colours[i % len(colours)]
        for XY in XYs:
            ax.plot(XY[:, 0], XY[:, 1], c=c, linewidth=2)
    
    ax.set_aspect('equal')
    ax.invert_yaxis()
    ax.axis("off")
    plt.savefig(output_image_path)
    plt.close(fig)

# Step 2: Classify and Smooth Contours
def classify_shape(cnt):
    epsilon = 0.04 * cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, epsilon, True)
    
    if len(approx) == 3:
        return 'triangle'
    elif len(approx) == 4:
        (x, y, w, h) = cv2.boundingRect(approx)
        aspect_ratio = w / float(h)
        if 0.95 <= aspect_ratio <= 1.05:
            return 'square'
        else:
            return 'rectangle'
    elif len(approx) == 5:
        return 'pentagon'
    elif len(approx) == 6:
        return 'hexagon'
    else:
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)
        circularity = 4 * np.pi * (area / (perimeter * perimeter))
        if 0.7 <= circularity <= 1.2:
            return 'circle'
        else:
            return 'other'

def smooth_contour(cnt, shape_type):
    if shape_type == 'triangle':
        epsilon = 0.05 * cv2.arcLength(cnt, True)
    elif shape_type == 'rectangle' or shape_type == 'square':
        epsilon = 0.02 * cv2.arcLength(cnt, True)
    elif shape_type == 'pentagon' or shape_type == 'hexagon':
        epsilon = 0.04 * cv2.arcLength(cnt, True)
    elif shape_type == 'circle':
        epsilon = 0.002 * cv2.arcLength(cnt, True)
    else: 
        epsilon = 0.03 * cv2.arcLength(cnt, True)
    return cv2.approxPolyDP(cnt, epsilon, True)

def process_image(input_image_path, output_image_path, output_csv_path):
    img = cv2.imread(input_image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    smoothed_contours = []
    for cnt in contours:
        shape_type = classify_shape(cnt)
        smoothed_contour = smooth_contour(cnt, shape_type)
        smoothed_contours.append(smoothed_contour)
    
    output_img = np.zeros_like(img)
    cv2.drawContours(output_img, smoothed_contours, -1, (255, 255, 255), 2)
    cv2.imwrite(output_image_path, output_img)
    
    # Saving smoothed contours to CSV
    with open(output_csv_path, 'w', newline='') as file:
        for contour in smoothed_contours:
            for point in contour:
                file.write(f"{point[0][0]},{point[0][1]}\n")

# Streamlit App
def main():
    st.title("Regularization and Smoothing")

    # Upload CSV file
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp_csv:
            tmp_csv.write(uploaded_file.read())
            tmp_csv_path = tmp_csv.name
        
        # Step 1: Plot CSV
        paths = read_csv(tmp_csv_path)
        output_png_path = tmp_csv_path.replace(".csv", ".png")
        plot(paths, output_png_path)
        
        # Step 2: Process Image
        output_jpg_path = tmp_csv_path.replace(".csv", ".jpg")
        output_smoothed_csv_path = tmp_csv_path.replace(".csv", "_smoothed.csv")
        process_image(output_png_path, output_jpg_path, output_smoothed_csv_path)
        
        # Display input and output images side by side
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(output_png_path, caption='Plotted Image from CSV', use_column_width=True)
        
        with col2:
            st.image(output_jpg_path, caption='Processed Image with Smoothed Contours', use_column_width=True)
        
        
        # Clean up temporary files
        os.remove(tmp_csv_path)
        os.remove(output_png_path)
        os.remove(output_jpg_path)
        os.remove(output_smoothed_csv_path)

if __name__ == "__main__":
    main()
