import cv2
import numpy as np
import time
import mediapipe as mp

# Mobile Camera Configuration
MOBILE_CAMERA_IP = "192.168.118.94"  # Change to your phone's IP
MOBILE_CAMERA_PORT = "8080"          # Default port for IP Webcam app
MOBILE_STREAM_URL = f"http://{MOBILE_CAMERA_IP}:{MOBILE_CAMERA_PORT}/video"

class LifeboatRescueTracker:
    def __init__(self):
        # Initialize MediaPipe Pose
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,  # 0=lite, 1=full, 2=heavy
            smooth_landmarks=True,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
        self.ws, self.hs = 800, 600  # Target resolution
        self.alert_active = False  # Track alert state
        self.last_alert_time = 0  # Time of last alert
        
        print("🚤 Lifeboat Rescue Body Tracker initialized")
        print(f"Mobile Camera URL: {MOBILE_STREAM_URL}")
        print()
        print("📋 Mobile Camera Setup Instructions:")
        print("1. Install 'IP Webcam' app on Android")
        print("2. Connect phone to same WiFi network")
        print("3. Start server in the app")
        print("4. Note the IP address shown")
        print("5. Update MOBILE_CAMERA_IP in this code")
        print("6. Use BACK camera for better body detection")
        print("-" * 50)
    
    def get_body_center(self, landmarks):
        """Calculate body center from pose landmarks"""
        if not landmarks:
            return None
            
        # Get key body points
        left_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        right_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        left_hip = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value]
        right_hip = landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value]
        
        # Calculate torso center
        shoulder_center_x = (left_shoulder.x + right_shoulder.x) / 2
        shoulder_center_y = (left_shoulder.y + right_shoulder.y) / 2
        hip_center_x = (left_hip.x + right_hip.x) / 2
        hip_center_y = (left_hip.y + right_hip.y) / 2
        
        # Body center (between shoulders and hips)
        body_center_x = (shoulder_center_x + hip_center_x) / 2
        body_center_y = (shoulder_center_y + hip_center_y) / 2
        
        return {
            'x': int(body_center_x * self.ws),
            'y': int(body_center_y * self.hs),
            'shoulders': {
                'x': int(shoulder_center_x * self.ws),
                'y': int(shoulder_center_y * self.hs)
            },
            'hips': {
                'x': int(hip_center_x * self.ws),
                'y': int(hip_center_y * self.hs)
            }
        }
    
    def connect_mobile_camera(self):
        """Connect to mobile camera stream"""
        try:
            # Try different URL formats for IP Webcam
            urls_to_try = [
                f"http://{MOBILE_CAMERA_IP}:{MOBILE_CAMERA_PORT}/video",      # IP Webcam standard
                f"http://{MOBILE_CAMERA_IP}:{MOBILE_CAMERA_PORT}/mjpg/video", # Alternative format
                f"http://{MOBILE_CAMERA_IP}:{MOBILE_CAMERA_PORT}/shot.jpg",   # Single frame mode
            ]
            
            for url in urls_to_try:
                print(f"📱 Trying to connect to: {url}")
                cap = cv2.VideoCapture(url)
                
                if cap.isOpened():
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        print(f"✅ Connected to mobile camera!")
                        return cap
                    else:
                        cap.release()
                
            print("❌ Could not connect to mobile camera")
            return None
            
        except Exception as e:
            print(f"❌ Mobile camera connection error: {e}")
            return None
    
    def run_tracking(self):
        """Main body tracking loop for lifeboat rescue"""
        print("\n🚤 Starting lifeboat rescue body tracking...")
        print("📱 Connecting to mobile camera...")
        
        # Connect to mobile camera
        cap = self.connect_mobile_camera()
        if cap is None:
            print("❌ Failed to connect to mobile camera")
            print("📋 Troubleshooting:")
            print("- Check if IP Webcam app is running on phone")
            print("- Verify phone and computer are on same WiFi")
            print("- Update MOBILE_CAMERA_IP in code")
            print("- Ensure BACK camera is selected in IP Webcam")
            return False
        
        print("✅ Mobile camera connected successfully!")
        print("🚶‍♂️ Body detection active - monitoring for people in water")
        print("Press 'q' in the tracking window to quit")
        print("-" * 50)
        
        frame_count = 0
        
        while True:
            # Get frame from mobile camera
            ret, frame = cap.read()
            
            if not ret or frame is None:
                print("📱 Waiting for mobile camera frame...")
                time.sleep(0.1)
                continue
            
            frame_count += 1
            if frame_count % 30 == 0:  # Print every 30 frames
                print(f"🚶‍♂️ Processed {frame_count} frames - monitoring for people")
            
            # Resize frame to target size
            frame = cv2.resize(frame, (self.ws, self.hs))
            
            # Convert BGR to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process the frame for pose detection
            results = self.pose.process(rgb_frame)
            
            # Create a semi-transparent overlay for alerts
            overlay = frame.copy()
            
            if results.pose_landmarks:
                # Draw pose landmarks on frame
                self.mp_drawing.draw_landmarks(
                    frame, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                    connection_drawing_spec=self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2))
                
                # Get body center
                body_center = self.get_body_center(results.pose_landmarks.landmark)
                
                if body_center:
                    bx, by = body_center['x'], body_center['y']
                    self.alert_active = True
                    self.last_alert_time = time.time()
                    
                    # Draw alert overlay (semi-transparent red)
                    cv2.rectangle(overlay, (0, 0), (self.ws, self.hs), (0, 0, 255), -1)
                    frame = cv2.addWeighted(overlay, 0.3, frame, 0.7, 0)
                    
                    # Draw body tracking visualization
                    cv2.circle(frame, (bx, by), 20, (0, 255, 0), -1)  # Body center
                    cv2.circle(frame, (bx, by), 100, (0, 255, 0), 3)  # Outer circle
                    
                    # Draw torso rectangle
                    shoulders = body_center['shoulders']
                    hips = body_center['hips']
                    cv2.rectangle(frame, 
                                 (shoulders['x'] - 50, shoulders['y'] - 20),
                                 (hips['x'] + 50, hips['y'] + 20),
                                 (255, 255, 0), 2)
                    
                    # Display alert message
                    cv2.putText(frame, "🚨 PERSON DETECTED!", (self.ws//4, self.hs//4),
                               cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
                    cv2.putText(frame, f"Position: ({bx}, {by})", (bx + 30, by - 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(frame, "🚶‍♂️ BODY DETECTED", (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    
            else:
                # No body detected
                if time.time() - self.last_alert_time > 3:  # Reset alert after 3 seconds
                    self.alert_active = False
                
                # Draw semi-transparent green overlay for safe state
                if not self.alert_active:
                    cv2.rectangle(overlay, (0, 0), (self.ws, self.hs), (0, 255, 0), -1)
                    frame = cv2.addWeighted(overlay, 0.2, frame, 0.8, 0)
                
                cv2.putText(frame, "🔍 NO PERSON DETECTED", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.putText(frame, "Scanning for people in water...", (10, 70),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                # Draw center crosshair
                center_x, center_y = self.ws // 2, self.hs // 2
                cv2.circle(frame, (center_x, center_y), 80, (128, 128, 128), 2)
                cv2.line(frame, (0, center_y), (self.ws, center_y), (128, 128, 128), 1)
                cv2.line(frame, (center_x, 0), (center_x, self.hs), (128, 128, 128), 1)
            
            # Enhanced UI elements
            # Header bar
            cv2.rectangle(frame, (0, 0), (self.ws, 50), (50, 50, 50), -1)
            cv2.putText(frame, "🚤 Lifeboat Rescue System", (10, 35),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
            
            # Status bar at bottom
            cv2.rectangle(frame, (0, self.hs - 70), (self.ws, self.hs), (50, 50, 50), -1)
            cv2.putText(frame, f"Frame: {frame_count}", (10, self.hs - 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame, f"📱 {MOBILE_CAMERA_IP}:{MOBILE_CAMERA_PORT}", (10, self.hs - 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            cv2.putText(frame, "MediaPipe Pose Detection", (self.ws - 250, self.hs - 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (128, 255, 128), 2)
            
            # Show frame
            cv2.imshow("🚤 Lifeboat Rescue Body Tracking", frame)
            
            # Check for quit key
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
        
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        print("✅ Lifeboat rescue tracking stopped")
        return True

def main():
    print("🚤 Lifeboat Rescue Body Tracking System")
    print("=" * 50)
    print("🔧 Configuration:")
    print(f"  Mobile Camera: {MOBILE_CAMERA_IP}:{MOBILE_CAMERA_PORT}")
    print("  Detection: Human body pose tracking")
    print()
    
    print("📋 Setup Instructions:")
    print("1. Install 'IP Webcam' app on Android phone")
    print("2. Connect phone to same WiFi as computer")
    print("3. Open IP Webcam app and tap 'Start Server'")
    print("4. IMPORTANT: Use BACK camera for better body detection")
    print("5. Note the IP address displayed (e.g., 192.168.1.100:8080)")
    print("6. Update MOBILE_CAMERA_IP and MOBILE_CAMERA_PORT in this code")
    print("7. Install: pip install mediapipe")
    print("8. Run this script")
    print()
    
    print("🚤 Rescue System Features:")
    print("- Detects human bodies in water using MediaPipe")
    print("- Displays prominent alert when person is detected")
    print("- Shows skeleton overlay for detected bodies")
    print("- Tracks body center (torso) for precise detection")
    print("- Optimized for IP Webcam mobile camera")
    print()
    
    # Start tracking
    tracker = LifeboatRescueTracker()
    
    try:
        success = tracker.run_tracking()
        if success:
            print("✅ Rescue tracking completed successfully")
        else:
            print("❌ Rescue tracking failed - check mobile camera connection")
    except KeyboardInterrupt:
        print("\n⚠️ Rescue tracking interrupted by user")
    except Exception as e:
        print(f"❌ Error: {e}")
        print("💡 Make sure to install: pip install mediapipe")

if __name__ == "__main__":
    main()