import cv2
import numpy as np
import time
import serial
from PIL import Image

class FinalGarbageSorter:
    def __init__(self, serial_port='/dev/ttyUSB0', baud_rate=9600):
        """Initialize the final garbage sorter with advanced detection"""
        self.cap = cv2.VideoCapture(0)
        self.serial_conn = None
        self.serial_port = serial_port
        self.baud_rate = baud_rate
        
        # 4 main categories
        self.classes = ['plastic', 'metal', 'paper', 'other']
        
        # Map classes to bin numbers (0-3 for 4 bins)
        self.class_to_bin = {
            'plastic': 0,  # Bin 0: Plastic
            'metal': 1,    # Bin 1: Metal
            'paper': 2,    # Bin 2: Paper
            'other': 3     # Bin 3: Other (glass, mixed, etc.)
        }
        
        # Initialize serial connection
        self.init_serial()
        
    def init_serial(self):
        """Initialize serial connection to Arduino"""
        try:
            self.serial_conn = serial.Serial(self.serial_port, self.baud_rate, timeout=1)
            time.sleep(2)  # Wait for Arduino to reset
            print(f"Serial connection established on {self.serial_port}")
        except Exception as e:
            print(f"Failed to connect to Arduino: {e}")
            print("Running in camera-only mode")
    
    def send_to_arduino(self, bin_number):
        """Send bin number to Arduino"""
        if self.serial_conn and self.serial_conn.is_open:
            try:
                self.serial_conn.write(f"{bin_number}\n".encode())
                print(f"Sent bin number {bin_number} to Arduino")
            except Exception as e:
                print(f"Error sending to Arduino: {e}")
    
    def detect_objects(self, frame):
        """Advanced detection using multiple features"""
        # Get frame dimensions
        height, width = frame.shape[:2]
        center_x, center_y = width // 2, height // 2
        
        # Analyze the center region
        roi_size = min(width, height) // 3
        roi = frame[center_y-roi_size:center_y+roi_size, center_x-roi_size:center_x+roi_size]
        
        if roi.size == 0:
            return []
        
        # Convert to different color spaces
        roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # Calculate multiple features
        features = self.extract_features(roi, roi_hsv, roi_gray)
        
        # Advanced classification logic
        class_name, confidence = self.classify_object(features)
        
        return [{
            'class': class_name,
            'confidence': confidence,
            'bbox': [center_x-roi_size, center_y-roi_size, center_x+roi_size, center_y+roi_size],
            'features': features
        }]
    
    def extract_features(self, roi, roi_hsv, roi_gray):
        """Extract comprehensive features for classification"""
        # Basic color statistics
        avg_bgr = np.mean(roi, axis=(0, 1))
        avg_hsv = np.mean(roi_hsv, axis=(0, 1))
        avg_gray = np.mean(roi_gray)
        
        # Texture analysis
        std_gray = np.std(roi_gray)
        color_variance = np.var(roi, axis=(0, 1))
        total_variance = np.sum(color_variance)
        
        # Shape analysis
        aspect_ratio = roi.shape[0] / roi.shape[1]
        
        # Edge analysis
        edges = cv2.Canny(roi_gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        
        # Contour analysis
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            contour_area = cv2.contourArea(largest_contour)
            contour_perimeter = cv2.arcLength(largest_contour, True)
            circularity = 4 * np.pi * contour_area / (contour_perimeter ** 2) if contour_perimeter > 0 else 0
        else:
            circularity = 0
        
        # Transparency analysis (for glass detection)
        transparency_score = self.analyze_transparency(roi_hsv)
        
        # Color distribution analysis
        color_distribution = self.analyze_color_distribution(roi_hsv)
        
        return {
            'avg_bgr': avg_bgr,
            'avg_hsv': avg_hsv,
            'avg_gray': avg_gray,
            'std_gray': std_gray,
            'total_variance': total_variance,
            'aspect_ratio': aspect_ratio,
            'edge_density': edge_density,
            'circularity': circularity,
            'transparency_score': transparency_score,
            'color_distribution': color_distribution
        }
    
    def analyze_transparency(self, roi_hsv):
        """Analyze transparency for glass detection"""
        # Glass is typically transparent with low saturation
        avg_saturation = np.mean(roi_hsv[:, :, 1])
        avg_value = np.mean(roi_hsv[:, :, 2])
        
        # High value + low saturation = transparent
        transparency = (avg_value / 255.0) * (1 - avg_saturation / 255.0)
        return transparency
    
    def analyze_color_distribution(self, roi_hsv):
        """Analyze color distribution patterns"""
        hue_hist = cv2.calcHist([roi_hsv], [0], None, [180], [0, 180])
        hue_hist = hue_hist.flatten() / np.sum(hue_hist)
        
        # Calculate color diversity
        color_diversity = np.sum(hue_hist > 0.01)  # Number of significant color peaks
        
        return {
            'hue_hist': hue_hist,
            'color_diversity': color_diversity
        }
    
    def classify_object(self, features):
        """Advanced classification using multiple features"""
        scores = {
            'plastic': 0,
            'metal': 0,
            'paper': 0,
            'other': 0
        }
        
        # Plastic detection
        if features['avg_gray'] < 120:  # Dark objects
            scores['plastic'] += 3
        if features['total_variance'] < 1500:  # Low variance
            scores['plastic'] += 2
        if features['aspect_ratio'] > 1.2:  # Bottle-shaped
            scores['plastic'] += 2
        if features['edge_density'] < 0.1:  # Smooth surface
            scores['plastic'] += 1
        if features['transparency_score'] > 0.3:  # Semi-transparent
            scores['plastic'] += 1
        
        # Metal detection
        if features['avg_hsv'][1] > 80:  # High saturation
            scores['metal'] += 3
        if features['total_variance'] > 2000:  # High variance
            scores['metal'] += 2
        if features['edge_density'] > 0.15:  # Many edges
            scores['metal'] += 2
        if features['std_gray'] > 40:  # High contrast
            scores['metal'] += 1
        if features['color_distribution']['color_diversity'] > 3:  # Multiple colors
            scores['metal'] += 1
        
        # Paper detection
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
        
        # Other detection (glass, mixed materials)
        if features['transparency_score'] > 0.5:  # Very transparent
            scores['other'] += 3
        if features['avg_gray'] > 100 and features['avg_gray'] < 150:  # Medium brightness
            scores['other'] += 2
        if features['total_variance'] > 1000 and features['total_variance'] < 2000:  # Medium variance
            scores['other'] += 2
        if features['circularity'] > 0.7:  # Circular objects
            scores['other'] += 1
        
        # Find best classification
        best_class = max(scores, key=scores.get)
        max_score = scores[best_class]
        total_score = sum(scores.values())
        
        # Calculate confidence
        confidence = max_score / total_score if total_score > 0 else 0.3
        
        # Adjust confidence based on feature strength
        if best_class == 'plastic' and features['avg_gray'] < 80:
            confidence = min(0.9, confidence + 0.2)
        elif best_class == 'metal' and features['avg_hsv'][1] > 100:
            confidence = min(0.9, confidence + 0.2)
        elif best_class == 'paper' and features['avg_gray'] > 180:
            confidence = min(0.9, confidence + 0.2)
        elif best_class == 'other' and features['transparency_score'] > 0.6:
            confidence = min(0.9, confidence + 0.2)
        
        # Ensure minimum confidence
        confidence = max(0.4, confidence)
        
        return best_class, confidence
    
    def draw_detections(self, frame, detections):
        """Draw detection results on the frame"""
        for detection in detections:
            bbox = detection['bbox']
            class_name = detection['class']
            confidence = detection['confidence']
            
            # Draw bounding box
            x1, y1, x2, y2 = bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw label with type and confidence
            label = f"{class_name}: {confidence:.2f}"
            cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Show bin assignment more prominently
            bin_number = self.class_to_bin.get(class_name, 0)
            bin_label = f"BIN {bin_number}"
            cv2.putText(frame, bin_label, (x1, y2+20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            
            # Add type and bin info in center of detection area
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            type_bin_text = f"{class_name.upper()} -> BIN {bin_number}"
            cv2.putText(frame, type_bin_text, (center_x-50, center_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        return frame
    
    def run(self):
        """Main loop for final garbage sorting"""
        print("=== Final Garbage Sorter (Advanced Detection) ===")
        print("Categories: plastic, metal, paper, other")
        print("Bins: 0=Plastic, 1=Metal, 2=Paper, 3=Other")
        print("Press 'q' to quit, 's' to sort manually")
        print("Auto-sorting when confidence > 0.6")
        print("Press 'd' to toggle debug mode")
        
        last_sort_time = 0
        sort_cooldown = 2.0  # Seconds between sorts
        auto_sort_enabled = True
        debug_mode = False
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Failed to grab frame")
                break
            
            # Mirror the frame for better user experience
            frame = cv2.flip(frame, 1)
            
            # Detect objects
            detections = self.detect_objects(frame)
            
            # Draw detections
            frame = self.draw_detections(frame, detections)
            
            # Add instructions
            cv2.putText(frame, "Press 'q' to quit, 's' to sort manually", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Show current detection prominently at top
            if detections:
                best_detection = max(detections, key=lambda x: x['confidence'])
                class_name = best_detection['class']
                confidence = best_detection['confidence']
                bin_number = self.class_to_bin.get(class_name, 0)
                
                # Draw background rectangle for better visibility
                cv2.rectangle(frame, (10, 40), (400, 80), (0, 0, 0), -1)
                cv2.rectangle(frame, (10, 40), (400, 80), (255, 255, 255), 2)
                
                # Show detection info
                detection_text = f"DETECTED: {class_name.upper()} -> BIN {bin_number}"
                cv2.putText(frame, detection_text, (20, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                confidence_text = f"Confidence: {confidence:.2f}"
                cv2.putText(frame, confidence_text, (20, 75), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                
                # Show debug features if available
                if debug_mode and 'features' in best_detection:
                    features = best_detection['features']
                    debug_y = 160
                    cv2.putText(frame, f"Gray: {features['avg_gray']:.1f}, Std: {features['std_gray']:.1f}", 
                               (10, debug_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
                    cv2.putText(frame, f"Variance: {features['total_variance']:.0f}, Edges: {features['edge_density']:.3f}", 
                               (10, debug_y+15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
                    cv2.putText(frame, f"Aspect: {features['aspect_ratio']:.2f}, Trans: {features['transparency_score']:.2f}", 
                               (10, debug_y+30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
            
            # Show auto-sort status
            status = "AUTO-SORT: ON" if auto_sort_enabled else "AUTO-SORT: OFF"
            cv2.putText(frame, status, (10, 100), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0) if auto_sort_enabled else (0, 0, 255), 2)
            
            # Show current time since last sort
            time_since_sort = time.time() - last_sort_time
            if time_since_sort < sort_cooldown:
                cv2.putText(frame, f"Cooldown: {sort_cooldown - time_since_sort:.1f}s", 
                           (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Auto-sort logic
            if auto_sort_enabled and detections and time_since_sort >= sort_cooldown:
                best_detection = max(detections, key=lambda x: x['confidence'])
                confidence = best_detection['confidence']
                
                if confidence > 0.6:  # Higher threshold for accuracy
                    class_name = best_detection['class']
                    bin_number = self.class_to_bin.get(class_name, 0)
                    print(f"AUTO-SORTING {class_name} (confidence: {confidence:.2f}) to bin {bin_number}")
                    self.send_to_arduino(bin_number)
                    last_sort_time = time.time()
                    
                    # Show auto-sort feedback
                    cv2.putText(frame, f"AUTO-SORTED: {class_name.upper()} -> BIN {bin_number}", 
                               (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # Display the frame
            cv2.imshow('Final Garbage Sorter', frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('s'):
                current_time = time.time()
                if current_time - last_sort_time >= sort_cooldown:
                    if detections:
                        best_detection = max(detections, key=lambda x: x['confidence'])
                        class_name = best_detection['class']
                        confidence = best_detection['confidence']
                        
                        if confidence > 0.5:
                            bin_number = self.class_to_bin.get(class_name, 0)
                            print(f"MANUAL SORT: {class_name} (confidence: {confidence:.2f}) to bin {bin_number}")
                            self.send_to_arduino(bin_number)
                            last_sort_time = current_time
                        else:
                            print(f"Confidence too low ({confidence:.2f}) for sorting")
                    else:
                        print("No object detected for sorting")
                else:
                    print("Please wait for cooldown to finish")
            elif key == ord('a'):  # Toggle auto-sort
                auto_sort_enabled = not auto_sort_enabled
                print(f"Auto-sort {'ENABLED' if auto_sort_enabled else 'DISABLED'}")
            elif key == ord('d'):  # Toggle debug mode
                debug_mode = not debug_mode
                print(f"Debug mode {'ENABLED' if debug_mode else 'DISABLED'}")
        
        # Cleanup
        self.cap.release()
        cv2.destroyAllWindows()
        if self.serial_conn:
            self.serial_conn.close()

def main():
    """Main function"""
    print("=== Final Garbage Sorter (Advanced Detection) ===")
    print("This system uses advanced feature analysis for accurate garbage classification")
    print("Categories: plastic, metal, paper, other")
    print("Bins: 0=Plastic, 1=Metal, 2=Paper, 3=Other")
    
    # Create and run the garbage sorter
    sorter = FinalGarbageSorter()
    sorter.run()

if __name__ == "__main__":
    main() 