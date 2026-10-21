// EMG-controlled BiDexHand — servo controller firmware
// Target: Arduino Uno/Nano or ESP32 + PCA9685 16-channel PWM driver (I2C, 0x40)
//
// The host (simulation/app.py --serial PORT, or a Raspberry Pi running the CNN-LSTM)
// sends one line per gesture change:   G<id>\n      id = 0..9 (table below)
// Optional direct servo command:        S<ch>,<deg>\n  ch = 0..15, deg = -90..90
//
// The firmware moves every servo towards its target with a speed limit
// (same 600 deg/s limit as the simulation) and reports "OK <id>" back.
// Servo channel order is identical to /hand_servo_input and simulation/emg_sim/bidexhand.py.
//
// Wiring: PCA9685 SDA/SCL -> MCU SDA/SCL, VCC -> 3.3/5 V logic,
//         V+ -> separate 5-6 V servo supply (>= 5 A for 16 micro servos), common GND.

#include <Wire.h>
#include <Adafruit_PWMServoDriver.h>

Adafruit_PWMServoDriver pwm = Adafruit_PWMServoDriver(0x40);

const uint8_t N_SERVO = 16;
const uint8_t N_GESTURE = 10;
const float SERVO_SPEED_DPS = 600.0f;     // deg/s
const uint16_t US_CENTER = 1500, US_PER_90 = 1000;
const uint32_t TICK_MS = 10;              // 100 Hz update loop
const uint32_t WATCHDOG_MS = 2000;        // no command for 2 s -> go to rest (fail-safe)

// Generated from simulation/emg_sim/bidexhand.py ALL_POSES (degrees)
const int8_t GESTURES[N_GESTURE][N_SERVO] = {
  {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},  // 0 rest
  {20, 0, 0, 0, 0, 0, 80, 80, 80, 80, 50, 85, 85, 85, 85, -60},  // 1 fist
  {10, 0, 0, 0, 0, 0, 60, 60, 60, 60, 30, 60, 60, 60, 60, -40},  // 2 flexion (close)
  {0, -8, -3, 3, 8, 0, -5, -5, -5, -5, 0, 0, 0, 0, 0, 0},  // 3 extension (open wide)
  {-1, 0, 0, 0, 0, 8, 45, 0, 0, 0, -20, 45, 0, 0, 0, -30},  // 4 pinch_index
  {0, 0, 0, 0, 0, 0, 45, 50, 0, 0, -11, 50, 55, 0, 0, -40},  // 5 pinch_middle (tripod)
  {0, 0, 0, 0, 0, -8, 0, 0, 55, 0, 0, 0, 0, 60, 0, -59},  // 6 pinch_ring
  {0, 0, 0, 0, 0, -24, 0, 0, 0, 60, 0, 0, 0, 0, 65, -70},  // 7 pinch_little
  {20, 0, 0, 0, 0, 0, 0, 80, 80, 80, 50, 0, 85, 85, 85, -60},  // 8 index_point
  {20, -8, 8, 0, 0, 0, 0, 0, 80, 80, 50, 0, 0, 85, 85, -60},  // 9 peace
};

float target[N_SERVO], actual[N_SERVO];
uint32_t lastTick = 0, lastCmd = 0;
char line[24];
uint8_t len = 0;

void writeServo(uint8_t ch, float deg) {
  deg = constrain(deg, -90.0f, 90.0f);
  uint16_t us = US_CENTER + (int16_t)(deg / 90.0f * US_PER_90);
  pwm.writeMicroseconds(ch, us);
}

void setGesture(uint8_t id) {
  if (id >= N_GESTURE) return;
  for (uint8_t i = 0; i < N_SERVO; i++) target[i] = GESTURES[id][i];
  Serial.print("OK ");
  Serial.println(id);
}

void handleLine(const char *s) {
  if (s[0] == 'G') {
    setGesture((uint8_t)atoi(s + 1));
    lastCmd = millis();
  } else if (s[0] == 'S') {
    int ch = atoi(s + 1);
    const char *c = strchr(s, ',');
    if (c && ch >= 0 && ch < N_SERVO) target[ch] = constrain(atof(c + 1), -90, 90);
    lastCmd = millis();
  }
}

void setup() {
  Serial.begin(115200);
  pwm.begin();
  pwm.setOscillatorFrequency(27000000);
  pwm.setPWMFreq(50);                       // standard analog servo frame
  for (uint8_t i = 0; i < N_SERVO; i++) { target[i] = actual[i] = 0; writeServo(i, 0); }
  Serial.println("EMG BiDexHand servo controller ready");
}

void loop() {
  while (Serial.available()) {
    char c = Serial.read();
    if (c == '\n' || c == '\r') {
      if (len) { line[len] = 0; handleLine(line); len = 0; }
    } else if (len < sizeof(line) - 1) {
      line[len++] = c;
    }
  }
  uint32_t now = millis();
  if (now - lastCmd > WATCHDOG_MS && lastCmd != 0) {   // host lost -> relax hand
    setGesture(0);
    lastCmd = 0;
  }
  if (now - lastTick >= TICK_MS) {
    float step = SERVO_SPEED_DPS * (now - lastTick) / 1000.0f;
    lastTick = now;
    for (uint8_t i = 0; i < N_SERVO; i++) {
      float e = target[i] - actual[i];
      actual[i] += constrain(e, -step, step);
      writeServo(i, actual[i]);
    }
  }
}
