import cv2
import numpy as np
import os
from pathlib import Path
import logging
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Set
from datetime import datetime
from ultralytics import YOLO
import time
import json

@dataclass
class CircleData:
    x: int
    y: int
    radius: int
    class_name: Optional[str] = None
    confidence: Optional[float] = None

class HoneycombAnalyzer:
    CLASS_COLORS = {
        'Capped': (255, 0, 0),     
        'Eggs': (0, 255, 0),       
        'Honey': (0, 0, 255),      
        'Larva': (255, 255, 0),    
        'Nectar': (255, 0, 255),   
        'Pollen': (0, 255, 255),   
        'Other': (128, 128, 128),  
        're_capped': (255, 128, 0) 
    }

    def __init__(self, model_path: str, save_dir: str):
        self.model = YOLO(model_path)
        self.save_dir = Path(save_dir)
        self.processed_files_path = self.save_dir / 'processed_files.json'
        self.processed_files = self.load_processed_files()
        self.setup_directories()
        self.setup_logging()

    def load_processed_files(self) -> Set[str]:
        """Load the set of processed files from JSON."""
        if self.processed_files_path.exists():
            with open(self.processed_files_path, 'r') as f:
                return set(json.load(f))
        return set()

    def save_processed_files(self):
        """Save the set of processed files to JSON."""
        with open(self.processed_files_path, 'w') as f:
            json.dump(list(self.processed_files), f)

    def setup_directories(self) -> None:
        """Create necessary output directories."""
        self.results_dir = self.save_dir / 'results'
        self.cropped_dir = self.save_dir / 'cropped_images'
        
        for directory in [self.results_dir, self.cropped_dir]:
            directory.mkdir(parents=True, exist_ok=True)

    def setup_logging(self) -> None:
        """Configure logging settings."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.save_dir / 'processing.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def detect_circles(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Detect circles in the image using HoughCircles."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=4.7,
            minDist=12,
            param1=80,
            param2=10,
            minRadius=6,
            maxRadius=12
        )
        
        return np.round(circles[0, :]).astype("int") if circles is not None else None

    def crop_circle(self, image: np.ndarray, circle: Tuple[int, int, int]) -> np.ndarray:
        """Crop and resize the circular region."""
        x, y, r = circle
        x1, y1 = max(x - r, 0), max(y - r, 0)
        x2, y2 = min(x + r, image.shape[1]), min(y + r, image.shape[0])
        cropped = image[y1:y2, x1:x2]
        return cv2.resize(cropped, (64, 64)) if cropped.size > 0 else np.array([])

    def classify_circles(self, cropped_images: List[np.ndarray]) -> List[Tuple[str, float]]:
        """
        Classify cropped circle images using YOLO.
        Returns a list of (class_name, confidence) tuples.
        """
        classifications = []
        
        try:
            results = self.model(cropped_images, verbose=False)
            
            if results is None:
                self.logger.error("YOLO model returned None.")
                return [('Unknown', 0.0)] * len(cropped_images)

            for i, r in enumerate(results):
                try:
                    # ตรวจสอบว่ามี probs หรือไม่ (สำหรับ YOLO classification)
                    if hasattr(r, 'probs') and r.probs is not None:
                        probs = r.probs.data.cpu().numpy()  # ดึงค่าความมั่นใจของแต่ละ class
                        max_conf_idx = probs.argmax()  # หาค่าที่มีความมั่นใจสูงสุด
                        confidence = float(probs[max_conf_idx])  # ดึงค่าความมั่นใจ
                        class_name = r.names[max_conf_idx] if max_conf_idx in r.names else "Unknown"

                        classifications.append((class_name, confidence))
                        self.logger.debug(f"Circle {i}: {class_name} with conf {confidence:.2f}")
                        continue
                    
                    # ถ้าใช้ bounding boxes
                    if r.boxes is not None and hasattr(r.boxes, 'conf') and hasattr(r.boxes, 'cls'):
                        boxes = r.boxes
                        confs = boxes.conf if boxes.conf is not None else []
                        cls = boxes.cls if boxes.cls is not None else []

                        if len(confs) > 0:
                            max_conf_idx = confs.argmax().item()
                            class_idx = int(cls[max_conf_idx].item())
                            confidence = float(confs[max_conf_idx].item())
                            class_name = r.names[class_idx] if class_idx in r.names else "Unknown"

                            classifications.append((class_name, confidence))
                            self.logger.debug(f"Circle {i}: {class_name} with conf {confidence:.2f}")
                            continue

                    self.logger.error(f"Result {i} has no valid bounding boxes or probabilities.")
                    classifications.append(('Unknown', 0.0))

                except Exception as e:
                    self.logger.error(f"Error processing result {i}: {str(e)}")
                    classifications.append(('Unknown', 0.0))

        except Exception as e:
            self.logger.error(f"Error during classification batch: {str(e)}")
        
        while len(classifications) < len(cropped_images):
            classifications.append(('Unknown', 0.0))
        
        return classifications


    def process_image(self, image_path: str) -> Tuple[List[CircleData], np.ndarray]:
        """Process a single image and return circle data and labeled image."""
        try:
            # Read image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not read image: {image_path}")

            # Detect circles
            detected_circles = self.detect_circles(image)
            if detected_circles is None:
                self.logger.info(f"No circles detected in {image_path}")
                return [], image

            # Process each circle
            circle_data = []
            cropped_images = []
            
            # Process circles
            for x, y, r in detected_circles:
                circle_data.append(CircleData(x=x, y=y, radius=r))
                cropped = self.crop_circle(image, (x, y, r))
                if cropped.size > 0:
                    cropped_images.append(cropped)

            # Classify circles if we have any
            if cropped_images:
                # Debug output
                self.logger.debug(f"Processing {len(cropped_images)} cropped images")
                
                # Get classifications
                classifications = self.classify_circles(cropped_images)
                
                # Debug output
                self.logger.debug(f"Got {len(classifications)} classifications")
                
                # Update circle data with classifications
                for circle, (class_name, confidence) in zip(circle_data, classifications):
                    circle.class_name = class_name
                    circle.confidence = confidence
                    # Debug output for each classification
                    self.logger.debug(f"Circle at ({circle.x}, {circle.y}): {class_name} {confidence:.2f}")

            # Draw and save results
            labeled_image = self.draw_results(image, circle_data)
            self.save_results(image_path, labeled_image, circle_data)

            return circle_data, labeled_image

        except Exception as e:
            self.logger.error(f"Error processing {image_path}: {str(e)}")
            return [], image

    def draw_results(self, image: np.ndarray, circles: List[CircleData]) -> np.ndarray:
        """Draw circles and labels on the image."""
        result_image = image.copy()
        
        for circle in circles:
            # Get color for the class, default to white only if class is Unknown
            if circle.class_name == 'Unknown':
                color = (255, 255, 255)  # White for unknown
            else:
                # Use the defined color or gray if class not in CLASS_COLORS
                color = self.CLASS_COLORS.get(circle.class_name, (128, 128, 128))
            
            # Draw circle
            cv2.circle(result_image, (circle.x, circle.y), circle.radius, color, 1)
            
            # ❌ ลบโค้ดที่วาด label ออกไป ❌
            # ไม่ต้องแสดงชื่อคลาสและค่าความมั่นใจแล้ว

        return result_image


    def save_results(self, image_path: str, labeled_image: np.ndarray, 
                        circle_data: List[CircleData]) -> None:
        """Save processing results."""
        image_name = Path(image_path).stem
        
        # Save labeled image
        cv2.imwrite(
            str(self.results_dir / f"{image_name}_labeled.jpg"),
            labeled_image
        )
        
        # Save circle data as numpy
        circle_dict_list = [vars(circle) for circle in circle_data]
        np.save(
            str(self.results_dir / f"{image_name}_circles.npy"),
            circle_dict_list
        )

        # Save classification results as text file
        with open(str(self.results_dir / f"{image_name}_classifications.txt"), 'w', encoding='utf-8') as f:
            f.write(f"Classification results for {image_name}\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Total circles detected: {len(circle_data)}\n\n")
            
            # Count class distribution
            class_counts = {}
            for circle in circle_data:
                if circle.class_name not in class_counts:
                    class_counts[circle.class_name] = 0
                class_counts[circle.class_name] += 1
            
            # Write summary
            f.write("Class Distribution:\n")
            for class_name, count in class_counts.items():
                f.write(f"{class_name}: {count} ({(count/len(circle_data)*100):.1f}%)\n")
            f.write("\n")
            
            # Write detailed results
            f.write("Detailed Classifications:\n")
            for i, circle in enumerate(circle_data, 1):
                f.write(f"Circle {i}: Position ({circle.x}, {circle.y}), "
                    f"Class: {circle.class_name}, "
                    f"Confidence: {circle.confidence:.2f}\n")

        # Generate and save statistics
        self.save_statistics(image_name, circle_data)

    def save_statistics(self, image_name: str, circle_data: List[CircleData]) -> None:
        """Generate and save analysis statistics."""
        stats = {
            'image_name': image_name,
            'total_cells': len(circle_data),
            'class_distribution': {},
            'average_confidence': 0.0,
            'timestamp': datetime.now().isoformat()
        }
        
        confidences = []
        for circle in circle_data:
            if circle.class_name and circle.class_name != 'Unknown':
                stats['class_distribution'][circle.class_name] = \
                    stats['class_distribution'].get(circle.class_name, 0) + 1
            if circle.confidence:
                confidences.append(circle.confidence)
        
        if confidences:
            stats['average_confidence'] = sum(confidences) / len(confidences)
        
        np.save(
            str(self.results_dir / f"{image_name}_statistics.npy"),
            stats
        )
    def monitor_folders(self, side_a_path: str, side_b_path: str, check_interval: int = 5):
        """
        Monitor two folders for new images and process them.
        
        Args:
            side_a_path: Path to side A folder
            side_b_path: Path to side B folder
            check_interval: Time in seconds between checks for new files
        """
        side_a = Path(side_a_path)
        side_b = Path(side_b_path)
        
        self.logger.info(f"Starting monitoring of folders:")
        self.logger.info(f"Side A: {side_a}")
        self.logger.info(f"Side B: {side_b}")

        while True:
            try:
                # Get all image files from both directories
                side_a_files = list(side_a.glob('*.[pj][np][g]'))
                side_b_files = list(side_b.glob('*.[pj][np][g]'))
                
                # Sort by creation time to get newest files first
                all_files = side_a_files + side_b_files
                all_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
                
                for image_path in all_files:
                    try:
                        # Check if file was already processed
                        if str(image_path) in self.processed_files:
                            continue
                        
                        self.logger.info(f"Processing new file: {image_path}")
                        
                        # Process the image
                        circle_data, _ = self.process_image(str(image_path))
                        
                        if circle_data:
                            self.logger.info(f"Found {len(circle_data)} circles in {image_path.name}")
                        else:
                            self.logger.warning(f"No circles found in {image_path.name}")
                        
                        # Add to processed files set
                        self.processed_files.add(str(image_path))
                        self.save_processed_files()
                        
                    except Exception as e:
                        self.logger.error(f"Error processing {image_path}: {str(e)}")
                        continue
                
                time.sleep(check_interval)
                
            except Exception as e:
                self.logger.error(f"Error during folder monitoring: {str(e)}")
                time.sleep(check_interval)


def main():
    """Main execution function."""
    # Configuration
    config = {
        'side_a_dir': "images/crop/side_A",
        'side_b_dir': "images/crop/side_B",
        'save_dir': "images/output",
        'model_path': "model_yolo/model_- 27 february 2025 13_21.pt"
    }
    
    # Initialize analyzer
    analyzer = HoneycombAnalyzer(config['model_path'], config['save_dir'])
    
    # Start monitoring folders
    analyzer.monitor_folders(config['side_a_dir'], config['side_b_dir'])

if __name__ == "__main__":
    main()