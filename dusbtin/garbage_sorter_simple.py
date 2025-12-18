import cv2
import numpy as np
import time
import serial
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import torchvision.models as models

class SimpleGarbageSorter:
    def __init__(self, serial_port='/dev/ttyUSB0', baud_rate=9600):
        """Initialize the simple garbage sorter with ImageNet pretrained model"""
        self.cap = cv2.VideoCapture(0)
        self.serial_conn = None
        self.serial_port = serial_port
        self.baud_rate = baud_rate
        
        # 4 main categories
        self.classes = ['plastic', 'metal', 'paper', 'other']
        
        # Transform for ImageNet pretrained models
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Map classes to bin numbers (0-3 for 4 bins)
        self.class_to_bin = {
            'plastic': 0,  # Bin 0: Plastic
            'metal': 1,    # Bin 1: Metal
            'paper': 2,    # Bin 2: Paper
            'other': 3     # Bin 3: Other (glass, mixed, etc.)
        }
        
        # Initialize the model
        self.model = self.load_simple_model()
        
        # Initialize serial connection
        self.init_serial()
        
    def load_simple_model(self):
        """Load a simple pretrained model"""
        print("Loading ImageNet pretrained model...")
        
        try:
            # Use ResNet18 (faster and lighter)
            model = models.resnet18(pretrained=True)
            
            # Replace the final layer for our 4 classes
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, len(self.classes))
            
            # Initialize the new layer with random weights
            nn.init.xavier_uniform_(model.fc.weight)
            nn.init.zeros_(model.fc.bias)
            
            model.eval()
            print("Model loaded successfully!")
            return model
            
        except Exception as e:
            print(f"Error loading model: {e}")
            return None
    
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
        """Detect objects using simple classification"""
        if self.model is None:
            return []
        
        # Get frame dimensions
        height, width = frame.shape[:2]
        center_x, center_y = width // 2, height // 2
        
        # Analyze the center region
        roi_size = min(width, height) // 3  # Larger ROI for better detection
        roi = frame[center_y-roi_size:center_y+roi_size, center_x-roi_size:center_x+roi_size]
        
        if roi.size == 0:
            return []
        
        # Convert ROI to PIL Image
        roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(roi_rgb)
        
        # Preprocess for model
        input_tensor = self.transform(pil_image).unsqueeze(0)
        
        # Get prediction
        with torch.no_grad():
            output = self.model(input_tensor)
            probabilities = torch.softmax(output, dim=1)
            predicted_class = torch.argmax(output, dim=1).item()
            confidence = probabilities[0][predicted_class].item()
        
        class_name = self.classes[predicted_class]
        
        return [{
            'class': class_name,
            'confidence': confidence,
            'bbox': [center_x-roi_size, center_y-roi_size, center_x+roi_size, center_y+roi_size],
            'probabilities': {self.classes[i]: prob.item() for i, prob in enumerate(probabilities[0])}
        }]
    
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
        """Main loop for simple garbage sorting"""
        print("=== Simple Garbage Sorter (ImageNet Pretrained) ===")
        print("Categories: plastic, metal, paper, other")
        print("Bins: 0=Plastic, 1=Metal, 2=Paper, 3=Other")
        print("Press 'q' to quit, 's' to sort manually")
        print("Auto-sorting when confidence > 0.5")
        print("Press 'd' to toggle debug mode")
        print("Note: This uses ImageNet weights, accuracy may vary")
        
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
                
                # Show debug probabilities if available
                if debug_mode and 'probabilities' in best_detection:
                    probs = best_detection['probabilities']
                    prob_text = f"Probs: P:{probs['plastic']:.2f} M:{probs['metal']:.2f} P:{probs['paper']:.2f} O:{probs['other']:.2f}"
                    cv2.putText(frame, prob_text, (20, 90), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
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
                
                if confidence > 0.5:  # Lower threshold for simple model
                    class_name = best_detection['class']
                    bin_number = self.class_to_bin.get(class_name, 0)
                    print(f"AUTO-SORTING {class_name} (confidence: {confidence:.2f}) to bin {bin_number}")
                    self.send_to_arduino(bin_number)
                    last_sort_time = time.time()
                    
                    # Show auto-sort feedback
                    cv2.putText(frame, f"AUTO-SORTED: {class_name.upper()} -> BIN {bin_number}", 
                               (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # Display the frame
            cv2.imshow('Simple Garbage Sorter', frame)
            
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
                        
                        if confidence > 0.3:
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
    print("=== Simple Garbage Sorter (ImageNet Pretrained) ===")
    print("This system uses ImageNet pretrained weights for immediate use")
    print("Categories: plastic, metal, paper, other")
    print("Bins: 0=Plastic, 1=Metal, 2=Paper, 3=Other")
    print("Note: Accuracy may vary as this is not specifically trained on garbage")
    
    # Create and run the garbage sorter
    sorter = SimpleGarbageSorter()
    sorter.run()

if __name__ == "__main__":
    main() 