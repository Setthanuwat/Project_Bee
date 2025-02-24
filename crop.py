import pupil_apriltags as apriltag
import cv2
import os
import numpy as np
from pathlib import Path

class ImageProcessor:
    def __init__(self, input_folder: str, output_folder: str):
        self.input_folder = Path(input_folder)
        self.output_folder = Path(output_folder)
        
        # Initialize AprilTag detector
        self.detector = apriltag.Detector(
            families='tag36h11',
            nthreads=4,
            quad_decimate=0.5,
            quad_sigma=0.5,
            refine_edges=True,
            debug=False
        )

    def set_folders(self, input_folder, output_folder):
        """
        Set input and output folders
        """
        self.input_folder = input_folder
        self.output_folder = output_folder
        
        # Create output folder if it doesn't exist
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)

    def get_latest_image(self):
        """ ค้นหารูปล่าสุดที่เข้าโฟลเดอร์ """
        image_files = [f for f in os.listdir(self.input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if not image_files:
            return None  # ไม่มีรูปในโฟลเดอร์
        latest_file = max(image_files, key=lambda f: os.path.getmtime(os.path.join(self.input_folder, f)))
        return os.path.join(self.input_folder, latest_file)

    def find_tag_centers(self, tags):
        """
        Find centers of all AprilTags
        """
        tag_centers = {}
        for tag in tags:
            tag_centers[tag.tag_id] = np.mean(tag.corners, axis=0)
        return tag_centers

    def draw_lines_on_frame(self, undistorted, tags):
        """
        Draw lines on frame around detected AprilTags and extract the region inside using perspective transform
        """
        frame_with_lines = undistorted.copy()
        tag_centers = self.find_tag_centers(tags)
        
        # Draw lines around each AprilTag
        for tag in tags:
            for j in range(4):
                cv2.line(frame_with_lines, tuple(tag.corners[j].astype(int)),
                        tuple(tag.corners[(j+1)%4].astype(int)), (0, 255, 0), 2)
            cv2.putText(frame_with_lines, str(tag.tag_id),
                        tuple(tag.center.astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        
        # Initialize warped image as None
        warped_image = None
        
        # Draw connecting lines between tags that follow their orientation
        if len(tag_centers) >= 2:
            # Get the two tag IDs
            tag_ids = list(tag_centers.keys())
            tag1_id, tag2_id = tag_ids[0], tag_ids[1]
            
            # Get their centers
            pt1 = tuple(tag_centers[tag1_id].astype(int))
            pt2 = tuple(tag_centers[tag2_id].astype(int))
            
            # Draw line between centers
            cv2.line(frame_with_lines, pt1, pt2, (0, 165, 255), 3)
            
            # Find the detected tags by ID
            tag1 = next((t for t in tags if t.tag_id == tag1_id), None)
            tag2 = next((t for t in tags if t.tag_id == tag2_id), None)
            
            if tag1 and tag2:
                # Calculate the orientation vector of tags
                tag1_orientation = tag1.corners[1] - tag1.corners[0]
                tag2_orientation = tag2.corners[1] - tag2.corners[0]
                
                # Normalize to unit vectors
                tag1_orientation = tag1_orientation / np.linalg.norm(tag1_orientation)
                tag2_orientation = tag2_orientation / np.linalg.norm(tag2_orientation)
                
                # Calculate perpendicular vectors for the frame height
                tag1_perp = np.array([-tag1_orientation[1], tag1_orientation[0]])
                tag2_perp = np.array([-tag2_orientation[1], tag2_orientation[0]])
                
                # Frame dimensions
                frame_width_offset = 100  # Adjust as needed
                frame_height = 1170       # Adjust as needed
                frame_width_offset2 = 80
                frame_height2 = 215
                
                # Calculate frame corners
                top_left = tag1.center + tag1_orientation * frame_width_offset - tag1_perp * frame_height/2
                top_right = tag2.center - tag2_orientation * frame_width_offset - tag2_perp * frame_height/2
                bottom_left = tag1.center + tag1_orientation * frame_width_offset2 - tag1_perp * frame_height2/2
                bottom_right = tag2.center - tag2_orientation * frame_width_offset2 - tag2_perp * frame_height2/2
                
                # Convert corners to integer arrays for drawing and perspective transform
                src_pts = np.array([
                    top_left.astype(int),
                    top_right.astype(int),
                    bottom_right.astype(int),
                    bottom_left.astype(int)
                ], dtype=np.float32)
                
                # Convert points to integers for drawing lines
                top_left_int = tuple(top_left.astype(int))
                top_right_int = tuple(top_right.astype(int))
                bottom_left_int = tuple(bottom_left.astype(int))
                bottom_right_int = tuple(bottom_right.astype(int))
                
                # Draw frame lines
                cv2.line(frame_with_lines, top_left_int, top_right_int, (0, 165, 255), 3)     # Top line
                cv2.line(frame_with_lines, bottom_left_int, bottom_right_int, (0, 165, 255), 3)  # Bottom line
                cv2.line(frame_with_lines, top_left_int, bottom_left_int, (0, 165, 255), 3)   # Left line
                cv2.line(frame_with_lines, top_right_int, bottom_right_int, (0, 165, 255), 3)  # Right line
                
                # Calculate width and height for the output image
                # Use the width of the top and the height of the left side
                width = int(np.linalg.norm(top_right - top_left))
                height = int(np.linalg.norm(bottom_left - top_left))
                
                # Define destination points for the perspective transform
                dst_pts = np.array([
                    [0, 0],
                    [width - 1, 0],
                    [width - 1, height - 1],
                    [0, height - 1]
                ], dtype=np.float32)
                
                # Get the perspective transform matrix
                M = cv2.getPerspectiveTransform(src_pts, dst_pts)
                
                # Apply the perspective transformation
                warped_image = cv2.warpPerspective(undistorted, M, (width, height))
                
                # Display the original image with lines and the warped result
        #         cv2.imshow("Frame with lines", frame_with_lines)
        #         if warped_image is not None:
        #             cv2.imshow("Warped region", warped_image)
        
        # cv2.waitKey(0)
        return frame_with_lines, warped_image

    def draw_lines_and_crop(self, undistorted, tags):
        """
        Draw lines and crop image based on AprilTag positions
        """
        frame_with_lines = undistorted.copy()
        
        if len(tags) >= 2:
            tag_centers = self.find_tag_centers(tags)
            frame_with_lines, cropped_image = self.draw_lines_on_frame(undistorted, tags)
            print("FIND 2 TAG")
            return frame_with_lines, cropped_image
        
        return undistorted, None

    def draw_lines_and_crop2(self, undistorted, tags):
        """
        Draw lines and crop image based on AprilTag positions
        """
        
        frame_with_lines = undistorted.copy()
        if len(tags) >= 2:
            tag_centers = self.find_tag_centers(tags)
            sorted_tag_ids = sorted(tag_centers.keys())
            pt1 = tuple(tag_centers[sorted_tag_ids[0]].astype(int))
            pt2 = tuple(tag_centers[sorted_tag_ids[1]].astype(int))
            
            try:
                # Calculate corner coordinates
                x_left_top = min(pt1[0], pt2[0]) 
                x_right_top = max(pt1[0], pt2[0])
                y_top = min(pt1[1], pt2[1])
                y_bottom = max(pt1[1], pt2[1])
                
                # Apply offsets
                y_top = y_top - 675
                y_bottom = y_bottom - 47
                x_left_top = x_left_top - 160
                x_right_top = x_right_top + 145
                
                # Ensure coordinates are within image bounds
                height, width = undistorted.shape[:2]
                y_top = max(0, y_top)
                y_bottom = min(height, y_bottom)
                x_left_top = max(0, x_left_top)
                x_right_top = min(width, x_right_top)
                
                if y_bottom > y_top and x_right_top > x_left_top:
                    cropped_image = undistorted[y_top:y_bottom, x_left_top:x_right_top]
                    return frame_with_lines, cropped_image
                else:
                    print("พื้นที่ครอปไม่ถูกต้อง")
                    return frame_with_lines, None
            except Exception as e:
                print(f"เกิดข้อผิดพลาดในการครอปภาพ: {e}")
                return frame_with_lines, None
                
        return undistorted, None

    def crop_with_fixed_coordinates(self, image):
        """
        Crop image using fixed coordinates
        """
        try:
            height, width = image.shape[:2]
            
            # Fixed coordinates for cropping
            x_left_top = 70
            x_right_top = 1230
            y_top = 0
            y_bottom = 580
            
            # Ensure coordinates are within image bounds
            x_left_top = max(0, x_left_top)
            x_right_top = min(width, x_right_top)
            y_top = max(0, y_top)
            y_bottom = min(height, y_bottom)
            
            if y_bottom > y_top and x_right_top > x_left_top:
                cropped_image = image[y_top:y_bottom, x_left_top:x_right_top]
                return cropped_image
            else:
                print("พิกัดที่กำหนดไว้ไม่ถูกต้อง")
                return None
        except Exception as e:
            print(f"เกิดข้อผิดพลาดในการครอปภาพด้วยพิกัดที่กำหนด: {e}")
            return None

    def process_latest_image(self):
        """ ประมวลผลเฉพาะรูปล่าสุด """
        if not self.input_folder or not self.output_folder:
            raise ValueError("ต้องกำหนด input และ output folder ก่อน")

        latest_image_path = self.get_latest_image()
        if latest_image_path is None:
            print("ไม่มีไฟล์รูปภาพในโฟลเดอร์")
            return
        
        filename = os.path.basename(latest_image_path)
        image = cv2.imread(latest_image_path)

        if image is None:
            print(f"ไม่สามารถอ่านรูป {filename} ได้")
            return

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        results = self.detector.detect(gray)

        output_filename = os.path.splitext(filename)[0] + '_cropped' + os.path.splitext(filename)[1]
        output_path = os.path.join(self.output_folder, output_filename)

        if len(results) > 1:
            print(f"{filename}: พบ {len(results)} AprilTag")
            frame_with_lines, cropped_image = self.draw_lines_and_crop(image, results)
            if cropped_image is not None:
                resized_image = cv2.resize(cropped_image, (1280, 720), interpolation=cv2.INTER_AREA)
                cv2.imwrite(output_path, resized_image)
                print(f"บันทึกภาพที่ครอปและปรับขนาดแล้วที่: {output_path}")
            else:
                print(f"{filename}: ไม่สามารถครอปภาพได้")
        else:
            print(f"{filename}: ไม่พบ AprilTag - ทำการครอปด้วยพิกัดที่กำหนดไว้")
            cropped_image = self.crop_with_fixed_coordinates(image)
            if cropped_image is not None:
                resized_image = cv2.resize(cropped_image, (1280, 720), interpolation=cv2.INTER_AREA)
                cv2.imwrite(output_path, resized_image)
                print(f"บันทึกภาพที่ครอปและปรับขนาดแล้วที่: {output_path}")
            else:
                print(f"{filename}: ไม่สามารถครอปภาพด้วยพิกัดที่กำหนดได้")

        print("เสร็จสิ้นการประมวลผลรูปล่าสุด")