import cv2
import numpy as np
import os

# สร้างโฟลเดอร์ชื่อ 'action_cam' ถ้ายังไม่มี
if not os.path.exists('action_cam'):
    os.makedirs('action_cam')

# โหลดข้อมูลการคาลิเบรทกล้อง
with np.load('calibration_data/CalibrationMatrix_college_cpt.npz') as data:
    camera_matrix = data['Camera_matrix']
    dist_coeffs = data['distCoeff']

# เปิดกล้องวิดีโอ
cap = cv2.VideoCapture(1)  # ค่า 0 หมายถึงกล้องตัวแรกที่เชื่อมต่อ
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1080)  # กำหนดความกว้าง
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)  # กำหนดความสูง

image_count = 0  # ตัวนับภาพที่บันทึก

while True:
    # อ่านภาพจากกล้อง
    ret, frame = cap.read()
    if not ret:
        print("ไม่สามารถอ่านภาพจากกล้องได้")
        break

    # ลบความบิดเบี้ยวของภาพโดยใช้ค่าการคาลิเบรท
    undistorted_frame = cv2.undistort(frame, camera_matrix, dist_coeffs)

    # แสดงภาพต้นฉบับและภาพที่ถูกลบความบิดเบี้ยว
    cv2.imshow('Original Frame', frame)
    cv2.imshow('Undistorted Frame', undistorted_frame)

    key = cv2.waitKey(1) & 0xFF
    
    # กด 'q' เพื่อออกจากโปรแกรม
    if key == ord('q'):
        break

    # กด 'f' เพื่อบันทึกภาพที่ถูกลบความบิดเบี้ยว
    if key == ord('f'): 
        image_count += 1
        file_name = f'action_cam/undistorted_image_{image_count}.jpg'
        
        # เช็คว่ามีไฟล์นี้อยู่หรือไม่
        while os.path.exists(file_name):
            image_count += 1
            file_name = f'action_cam/undistorted_image_{image_count}.jpg'
        
        cv2.imwrite(file_name, undistorted_frame)
        print(f"บันทึกภาพ: {file_name}")

# ปิดการใช้งานกล้องและหน้าต่างทั้งหมด
cap.release()
cv2.destroyAllWindows()
