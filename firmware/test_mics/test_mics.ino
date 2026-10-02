/* ---------------------------------------------------------------------
   test_mics.ino -- DIAGNOSTIC ONLY. Upload this first.

   Prints the loudness of all four microphones as text, ten times a
   second, so you can read it in the Arduino Serial Monitor with no
   Python involved at all.

   HOW TO USE
     1. Upload.
     2. Tools -> Serial Monitor, set the baud rate to 115200.
     3. Make a noise. All four bars should move.
     4. THE TAP TEST: scratch a fingernail on top of ONE microphone.
        Exactly one bar should jump. That tells you which Teensy channel
        that physical microphone is. Write the channel number on the board
        next to it with a marker pen, then repeat for the other three.

   WHAT THE NUMBERS MEAN
     0.0000        silent. Dead mic, bad solder joint, or L/R on the
                   wrong rail.
     0.0001-0.001  room tone. Normal when nobody is making a noise.
     0.01-0.2      someone talking or a speaker playing.
     ~1.0          clipping. Move the source further away.

   WIRING (Teensy 4.0 -- these pins are FIXED by the hardware, you
   cannot move them to different pins):

     all four mics:  VDD -> 3.3V    SCK -> pin 21  (bit clock)
                     GND -> GND     WS  -> pin 20  (word select)

     mic 0:          SD -> pin 8    L/R -> GND
     mic 1:          SD -> pin 8    L/R -> 3.3V
     mic 2:          SD -> pin 6    L/R -> GND
     mic 3:          SD -> pin 6    L/R -> 3.3V

   WARNING: Teensy 4.0 pins are NOT 5 V tolerant. The INMP441 must be
   powered from 3.3 V. 5 V will destroy both the microphone and the pin.
   --------------------------------------------------------------------- */

#include <Audio.h>

AudioInputI2SQuad mics;  // 4 channels, ONE clock domain
AudioAnalyzeRMS rms0, rms1, rms2, rms3;

AudioConnection c0(mics, 0, rms0, 0);
AudioConnection c1(mics, 1, rms1, 0);
AudioConnection c2(mics, 2, rms2, 0);
AudioConnection c3(mics, 3, rms3, 0);

static void bar(float v) {
  // logarithmic-ish bar so quiet signals are still visible
  int n = 0;
  if (v > 0.00002f) n = (int)(30.0f * (log10f(v) + 4.0f) / 4.0f);
  if (n < 0) n = 0;
  if (n > 30) n = 30;
  for (int i = 0; i < 30; i++) Serial.print(i < n ? '#' : '.');
}

void setup() {
  AudioMemory(60);
  Serial.begin(115200);
  delay(1500);
  Serial.println("4-channel INMP441 level meter");
  Serial.println("ch0 = pin8/LR-low   ch1 = pin8/LR-high");
  Serial.println("ch2 = pin6/LR-low   ch3 = pin6/LR-high");
  Serial.println();
}

void loop() {
  static uint32_t last = 0;
  if (millis() - last < 100) return;
  last = millis();

  float v[4] = { rms0.read(), rms1.read(), rms2.read(), rms3.read() };

  int loudest = 0;
  for (int i = 1; i < 4; i++) if (v[i] > v[loudest]) loudest = i;

  for (int i = 0; i < 4; i++) {
    Serial.print("ch");
    Serial.print(i);
    Serial.print(" [");
    bar(v[i]);
    Serial.print("] ");
    Serial.print(v[i], 5);
    if (v[i] < 0.00002f) Serial.print(" *** SILENT ***");
    else if (i == loudest && v[i] > 0.004f) Serial.print(" <== LOUDEST");
    Serial.println();
  }
  Serial.println();
}
