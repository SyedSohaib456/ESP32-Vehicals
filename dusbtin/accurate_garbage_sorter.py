import cv2
import numpy as np
import time
import serial

class AccurateGarbageSorter:
    def __init__(self, serial_port='/dev/ttyUSB0', baud_rate=9600):
        self.cap = cv2.VideoCapture(0)
        self.serial_conn = None
        self.serial_port = serial_port
        self.baud_rate = baud_rate
        
        # 4 categories
        self.classes = ['plastic', 'metal', 'paper', 'other']
        self.class_to_bin = {
            'plastic': 0, 'metal': 1, 'paper': 2, 'other': 3
        }
        
        self.init_serial()
    
    def init_serial(self):
        try:
            self.serial_conn = serial.Serial(self.serial_port, self.baud_rate, timeout=1)
            time.sleep(2)
            print(f"Connected to Arduino on {self.serial_port}")
        except Exception as e:
            print(f"Arduino connection failed: {e}")
    
    def send_to_arduino(self, bin_number):
        if self.serial_conn and self.serial_conn.is_open:
            try:
                self.serial_conn.write(f"{bin_number}\n".encode())
                print(f"Sent to bin {bin_number}")
            except Exception as e:
                print(f"Arduino error: {e}")
    
    def analyze_object(self, frame):
        height, width = frame.shape[:2]
        center_x, center_y = width // 2, height // 2
        
        # Analyze center region
        roi_size = min(width, height) // 4
        roi = frame[center_y-roi_size:center_y+roi_size, center_x-roi_size:center_x+roi_size]
        
        if roi.size == 0:
            return None
        
        # Convert color spaces
        roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # Calculate features
        avg_bgr = np.mean(roi, axis=(0, 1))
        avg_hsv = np.mean(roi_hsv, axis=(0, 1))
        avg_gray = np.mean(roi_gray)
        std_gray = np.std(roi_gray)
        color_variance = np.var(roi, axis=(0, 1))
        total_variance = np.sum(color_variance)
        
        # Edge analysis
        edges = cv2.Canny(roi_gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        
        # Shape analysis
        aspect_ratio = roi.shape[0] / roi.shape[1]
        
        # Transparency analysis
        transparency = (avg_hsv[2] / 255.0) * (1 - avg_hsv[1] / 255.0)
        
        return {
            'avg_bgr': avg_bgr,
            'avg_hsv': avg_hsv,
            'avg_gray': avg_gray,
            'std_gray': std_gray,
            'total_variance': total_variance,
            'edge_density': edge_density,
            'aspect_ratio': aspect_ratio,
            'transparency': transparency
        }
    
    def classify_object(self, features):
        scores = {'plastic': 0, 'metal': 0, 'paper': 0, 'other': 0}
        
        # Plastic detection (bottles, containers)
        if features['avg_gray'] < 120:  # Dark objects
            scores['plastic'] += 3
        if features['total_variance'] < 1500:  # Low variance
            scores['plastic'] += 2
        if features['aspect_ratio'] > 1.2:  # Bottle-shaped
            scores['plastic'] += 2
        if features['edge_density'] < 0.1:  # Smooth surface
            scores['plastic'] += 1
        if features['transparency'] > 0.3:  # Semi-transparent
            scores['plastic'] += 1
        
        # Metal detection (cans, metal objects)
        if features['avg_hsv'][1] > 80:  # High saturation
            scores['metal'] += 3
        if features['total_variance'] > 2000:  # High variance
            scores['metal'] += 2
        if features['edge_density'] > 0.15:  # Many edges
            scores['metal'] += 2
        if features['std_gray'] > 40:  # High contrast
            scores['metal'] += 1
        
        # Paper detection (cardboard, paper)
        if features['avg_gray'] > 150:  # Bright/white
            scores['paper'] += 3
        if features['total_variance'] < 1000:  # Low variance
            scores['paper'] += 2
        if features['edge_density'] < 0.08:  # Few edges
            scores['paper'] += 2
        if features['aspect_ratio'] < 1.3:  # Not bottle-shaped
            scores['paper'] += 1
        if features['std_gray'] < 30:  # Uniform texture
            scores['paper'] += 1
        
        # Other detection (glass, mixed)
        if features['transparency'] > 0.5:  # Very transparent
            scores['other'] += 3
        if 100 < features['avg_gray'] < 150:  # Medium brightness
            scores['other'] += 2
        if 1000 < features['total_variance'] < 2000:  # Medium variance
            scores['other'] += 2
        
        # Find best classification
        best_class = max(scores, key=scores.get)
        max_score = scores[best_class]
        total_score = sum(scores.values())
        confidence = max_score / total_score if total_score > 0 else 0.3
        
        # Boost confidence for strong indicators
        if best_class == 'plastic' and features['avg_gray'] < 80:
            confidence = min(0.9, confidence + 0.2)
        elif best_class == 'metal' and features['avg_hsv'][1] > 100:
            confidence = min(0.9, confidence + 0.2)
        elif best_class == 'paper' and features['avg_gray'] > 180:
            confidence = min(0.9, confidence + 0.2)
        
        confidence = max(0.4, confidence)
        return best_class, confidence
    
    def run(self):
        print("=== Accurate Garbage Sorter ===")
        print("Categories: plastic, metal, paper, other")
        print("Bins: 0=Plastic, 1=Metal, 2=Paper, 3=Other")
        print("Press 'q' to quit, 's' to sort manually")
        print("Auto-sorting when confidence > 0.6")
        
        last_sort_time = 0
        sort_cooldown = 2.0
        auto_sort = True
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            
            # Analyze object
            features = self.analyze_object(frame)
            
            if features:
                class_name, confidence = self.classify_object(features)
                bin_number = self.class_to_bin[class_name]
                
                # Draw detection box
                height, width = frame.shape[:2]
                center_x, center_y = width // 2, height // 2
                roi_size = min(width, height) // 4
                
                cv2.rectangle(frame, 
                             (center_x-roi_size, center_y-roi_size),
                             (center_x+roi_size, center_y+roi_size),
                             (0, 255, 0), 2)
                
                # Show detection info
                cv2.rectangle(frame, (10, 10), (400, 80), (0, 0, 0), -1)
                cv2.rectangle(frame, (10, 10), (400, 80), (255, 255, 255), 2)
                
                cv2.putText(frame, f"DETECTED: {class_name.upper()}", 
                           (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(frame, f"BIN: {bin_number} | Confidence: {confidence:.2f}", 
                           (20, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                
                # Auto-sort
                current_time = time.time()
                if auto_sort and confidence > 0.6 and current_time - last_sort_time >= sort_cooldown:
                    print(f"AUTO-SORT: {class_name} (conf: {confidence:.2f}) -> bin {bin_number}")
                    self.send_to_arduino(bin_number)
                    last_sort_time = current_time
                    
                    cv2.putText(frame, f"AUTO-SORTED TO BIN {bin_number}", 
                               (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            # Show controls
            cv2.putText(frame, "Press 'q' to quit, 's' to sort manually", 
                       (10, height-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            cv2.imshow('Accurate Garbage Sorter', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                current_time = time.time()
                if current_time - last_sort_time >= sort_cooldown and features:
                    class_name, confidence = self.classify_object(features)
                    if confidence > 0.5:
                        bin_number = self.class_to_bin[class_name]
                        print(f"MANUAL SORT: {class_name} -> bin {bin_number}")
                        self.send_to_arduino(bin_number)
                        last_sort_time = current_time
                    else:
                        print(f"Confidence too low: {confidence:.2f}")
                else:
                    print("Please wait for cooldown or no object detected")
        
        self.cap.release()
        cv2.destroyAllWindows()
        if self.serial_conn:
            self.serial_conn.close()

if __name__ == "__main__":
    sorter = AccurateGarbageSorter()
    sorter.run() 