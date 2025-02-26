import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
import os


def detect_wooden_frame_with_canny(image_path):
    if not os.path.exists(image_path):
        print(f"ไม่พบไฟล์ภาพที่: {image_path}")
        return None, None, None, None, None, None

    frame = cv2.imread(image_path)
    
    if frame is None:
        print(f"ไม่สามารถอ่านไฟล์ภาพจาก: {image_path}")
        return None, None, None, None, None, None

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 100, 180)
    
    # Morphological filtering to reduce unwanted edges
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    edges = cv2.dilate(edges, kernel, iterations=10)
    edges = cv2.erode(edges, kernel, iterations=10)
    
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    frame_contour = None
    max_area = 0
    for contour in contours:
        area = cv2.contourArea(contour)
        x, y, w, h = cv2.boundingRect(contour)
        if area > max_area and w > 200 and h > 200:
            max_area = area
            frame_contour = contour
            
    if frame_contour is not None:
        x, y, w, h = cv2.boundingRect(frame_contour)
        padding = 1
        x = max(x - padding, 0)
        y = max(y - padding, 0)
        w = min(w + 2 * padding, frame.shape[1] - x)
        h = min(h + 2 * padding, frame.shape[0] - y)
        
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        bee_frame = frame[y:y + h, x:x + w]
        return bee_frame, frame, x, y, w, h
    else:
        frame = cv2.resize(frame, (800, 800))
        plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        plt.title('Detected Wooden Frame')
        plt.axis('off')
        plt.show()
        return None, frame, None, None, None, None

def detect_circles(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 250, apertureSize=3)
    
    accumulator_threshold = 55
    min_distance = 12
    
    circle_counts_by_radius = []
    for i in range(5, 50, 5):
        min_radius = i + 1
        max_radius = i + 5
        print(f"Detecting circles with radius range: {min_radius} to {max_radius}")

        circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, dp=1.2, minDist=min_distance,
                                   param1=145, param2=accumulator_threshold,
                                   minRadius=min_radius, maxRadius=max_radius)

        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            circle_counts_by_radius.append((i, len(circles)))
            print(f"Detected {len(circles)} circles with radius in range {min_radius}-{max_radius}")
        else:
            circle_counts_by_radius.append((i, 0))

    most_frequent_radius = max(circle_counts_by_radius, key=lambda x: x[1])[0]
    print(f"Most frequent radius detected: {most_frequent_radius}")

    min_radius_final = int(most_frequent_radius - 0.1 * most_frequent_radius)
    max_radius_final = int(most_frequent_radius + 0.1 * most_frequent_radius)

    print(f"Running CHT again with minRadius = {min_radius_final} and maxRadius = {max_radius_final}")
    final_circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, dp=1.2, minDist=min_distance,
                                     param1=145, param2=accumulator_threshold,
                                     minRadius=min_radius_final, maxRadius=max_radius_final)

    if final_circles is not None:
        final_circles = np.round(final_circles[0, :]).astype("int")
        for (x, y, r) in final_circles:
            cv2.circle(image, (x, y), r, (0, 255, 0), 4)
            cv2.rectangle(image, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)

    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title('Detected Circles')
    plt.axis('off')
    plt.show()

start_time = time.time()
bee_frame, detected_frame, x, y, w, h = detect_wooden_frame_with_canny("F5.2.jpeg")
if detected_frame is not None:
    detect_circles(detected_frame)
end_time = time.time()

print(f"Processing complete in {end_time - start_time:.2f} seconds.")

