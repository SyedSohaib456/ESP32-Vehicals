#include <ESP8266WiFi.h>
#include <ESP8266WebServer.h>

// Function prototypes
void handleRoot();
void handleForward();
void handleBackward();
void handleLeft();
void handleRight();
void handleStop();
void handleSpeed();
void handleEmergencyLight();

// Access Point credentials
const char* ap_ssid = "RESCUE_BOAT_001";
const char* ap_password = "RESCUE2025";

// Create WebServer object on port 80
ESP8266WebServer server(80);

// IBT2 Motor Driver pins for Left Motor
const int RPWM_LEFT = 5;   // GPIO5  - RPWM for left motor
const int LPWM_LEFT = 4;   // GPIO4  - LPWM for left motor

// IBT2 Motor Driver pins for Right Motor
const int RPWM_RIGHT = 14; // GPIO14 - RPWM for right motor
const int LPWM_RIGHT = 12; // GPIO12 - LPWM for right motor

// Emergency Light Pin
const int emergencyLightPin = 16;  // GPIO16 for emergency light

// PWM properties
const int PWM_MAX = 1023;  // ESP8266 PWM range is 0-1023
int motorSpeed = 800;      // Default motor speed (0-1023)

// Emergency light state
bool emergencyLightState = false;

void setup() {
  Serial.begin(115200);
  
  // Configure pins as outputs
  pinMode(RPWM_LEFT, OUTPUT);
  pinMode(LPWM_LEFT, OUTPUT);
  pinMode(RPWM_RIGHT, OUTPUT);
  pinMode(LPWM_RIGHT, OUTPUT);
  
  // Set emergency light pin as output
  pinMode(emergencyLightPin, OUTPUT);
  digitalWrite(emergencyLightPin, LOW);
  
  // Set PWM range for ESP8266
  analogWriteRange(PWM_MAX);
  analogWriteFreq(5000); // Set PWM frequency to 5KHz
  
  // Stop motors initially
  stopMotors();
  
  // Create Access Point
  WiFi.softAP(ap_ssid, ap_password);
  
  Serial.println("===============================================");
  Serial.println("🚨 RESCUE BOAT CONTROL SYSTEM ONLINE");
  Serial.println("===============================================");
  Serial.print("Network Name: ");
  Serial.println(ap_ssid);
  Serial.print("Network Password: ");
  Serial.println(ap_password);
  Serial.print("Access Point IP: ");
  Serial.println(WiFi.softAPIP());
  Serial.println("===============================================");
  Serial.println("📱 Connect your device to this network");
  Serial.println("🌐 Then open browser and go to: 192.168.4.1");
  Serial.println("===============================================");
  
  // Define web server routes
  server.on("/", handleRoot);
  server.on("/forward", handleForward);
  server.on("/backward", handleBackward);
  server.on("/left", handleLeft);
  server.on("/right", handleRight);
  server.on("/stop", handleStop);
  server.on("/speed", handleSpeed);
  server.on("/emergency", handleEmergencyLight);
  
  // Start server
  server.begin();
}

void loop() {
  server.handleClient();
  yield(); // Allow ESP8266 to handle background tasks
}

// Motor control functions
void stopMotors() {
  analogWrite(RPWM_LEFT, 0);
  analogWrite(LPWM_LEFT, 0);
  analogWrite(RPWM_RIGHT, 0);
  analogWrite(LPWM_RIGHT, 0);
}

void moveForward() {
  // Left motor forward
  analogWrite(RPWM_LEFT, motorSpeed);
  analogWrite(LPWM_LEFT, 0);
  
  // Right motor forward
  analogWrite(RPWM_RIGHT, motorSpeed);
  analogWrite(LPWM_RIGHT, 0);
}

void moveBackward() {
  // Left motor backward
  analogWrite(RPWM_LEFT, 0);
  analogWrite(LPWM_LEFT, motorSpeed);
  
  // Right motor backward
  analogWrite(RPWM_RIGHT, 0);
  analogWrite(LPWM_RIGHT, motorSpeed);
}

void turnLeft() {
  // Left motor backward
  analogWrite(RPWM_LEFT, 0);
  analogWrite(LPWM_LEFT, motorSpeed);
  
  // Right motor forward
  analogWrite(RPWM_RIGHT, motorSpeed);
  analogWrite(LPWM_RIGHT, 0);
}

void turnRight() {
  // Left motor forward
  analogWrite(RPWM_LEFT, motorSpeed);
  analogWrite(LPWM_LEFT, 0);
  
  // Right motor backward
  analogWrite(RPWM_RIGHT, 0);
  analogWrite(LPWM_RIGHT, motorSpeed);
}

// Web server handlers
void handleRoot() {
  String html = R"html(
  <!DOCTYPE html>
  <!-- [Your HTML code remains unchanged] -->
  )html";
  server.send(200, "text/html", html);
}

void handleForward() {
  moveForward();
  server.send(200, "text/plain", "Moving Forward");
  Serial.println("Moving Forward");
}

void handleBackward() {
  moveBackward();
  server.send(200, "text/plain", "Moving Backward");
  Serial.println("Moving Backward");
}

void handleLeft() {
  turnLeft();
  server.send(200, "text/plain", "Turning Left");
  Serial.println("Turning Left");
}

void handleRight() {
  turnRight();
  server.send(200, "text/plain", "Turning Right");
  Serial.println("Turning Right");
}

void handleStop() {
  stopMotors();
  server.send(200, "text/plain", "Stopped");
  Serial.println("Stopped");
}

void handleSpeed() {
  if (server.hasArg("value")) {
    int newSpeed = server.arg("value").toInt();
    // Convert from 0-255 range to 0-1023 range
    motorSpeed = map(newSpeed, 0, 255, 0, PWM_MAX);
    motorSpeed = constrain(motorSpeed, 0, PWM_MAX);
    server.send(200, "text/plain", "Speed set to: " + String(motorSpeed));
    Serial.println("Speed set to: " + String(motorSpeed));
  } else {
    server.send(400, "text/plain", "Missing speed value");
  }
}

void handleEmergencyLight() {
  emergencyLightState = !emergencyLightState;
  digitalWrite(emergencyLightPin, emergencyLightState ? HIGH : LOW);
  
  if (emergencyLightState) {
    server.send(200, "text/plain", "Emergency Lights ON");
    Serial.println("Emergency Lights ON");
  } else {
    server.send(200, "text/plain", "Emergency Lights OFF");
    Serial.println("Emergency Lights OFF");
  }
}