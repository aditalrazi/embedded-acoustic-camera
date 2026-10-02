/* ---------------------------------------------------------------------
   four_mic_stream.ino -- the real firmware.

   Streams four sample-locked microphone channels to the computer over USB.

   THE WHOLE POINT OF THIS PROJECT IS IN ONE OBJECT: AudioInputI2SQuad.
   It makes the Teensy generate ONE bit clock and ONE word-select signal
   that all four microphones share. Every microphone therefore converts
   its sample at exactly the same instant, driven by the same crystal.
   That is what the brief means by "a common sampling clock", and it is
   why you cannot do this project with two USB microphones: two separate
   crystals drift apart by tens of microseconds per second, and tens of
   microseconds is the entire signal you are trying to measure.

   WIRE FORMAT -- one frame, 1028 bytes, 344.5 frames per second:

     byte 0      0xAA  \
     byte 1      0x55  |  4-byte header
     byte 2-3    sequence number, uint16, little-endian, wraps at 65535
     byte 4      128 samples of channel 0, int16 little-endian (256 bytes)
     ...         128 samples of channel 1 (256 bytes)
     ...         128 samples of channel 2 (256 bytes)
     ...         128 samples of channel 3 (256 bytes)

   Why a sync marker? USB is a stream of bytes with no message
   boundaries. Lose one byte and every channel afterwards is shifted, and
   your angles turn to nonsense with no obvious cause. The marker lets
   the Python reader find the start of a frame again by itself. The
   sequence number lets you SEE dropped frames instead of guessing.

   Total data rate: 4 ch x 44100 Hz x 2 bytes = 353 kB/s, which is
   nothing for Teensy 4's USB.

   NOTE: this firmware sends BINARY, not text. If you open the Arduino
   Serial Monitor while it is running you will see meaningless symbols.
   That is correct and expected -- close it and use the Python tools.
   --------------------------------------------------------------------- */

#include <Audio.h>

AudioInputI2SQuad mics;
AudioRecordQueue q0, q1, q2, q3;

AudioConnection c0(mics, 0, q0, 0);
AudioConnection c1(mics, 1, q1, 0);
AudioConnection c2(mics, 2, q2, 0);
AudioConnection c3(mics, 3, q3, 0);

const int LED = 13;
uint16_t seq = 0;

void setup() {
  pinMode(LED, OUTPUT);
  AudioMemory(300);       // plenty of headroom for four queues
  Serial.begin(2000000);  // baud is ignored on USB, it always runs full speed
  q0.begin(); q1.begin(); q2.begin(); q3.begin();
}

void loop() {
  // If the computer stops reading, the queues back up. Rather than let
  // them overflow at different moments and silently break channel
  // alignment, notice the backlog and clear all four together. Clearing
  // together loses audio but never loses synchronisation, and lost audio
  // is visible in the sequence numbers.
  if (q0.available() > 30) {
    q0.clear(); q1.clear(); q2.clear(); q3.clear();
    return;
  }

  // Emit a frame ONLY when all four blocks are ready. Never send a
  // partial set -- that is exactly how channels get out of step.
  if (q0.available() && q1.available() && q2.available() && q3.available()) {
    uint8_t hdr[4];
    hdr[0] = 0xAA;
    hdr[1] = 0x55;
    hdr[2] = (uint8_t)(seq & 0xFF);
    hdr[3] = (uint8_t)(seq >> 8);
    seq++;

    Serial.write(hdr, 4);

    int16_t *b;
    b = q0.readBuffer(); Serial.write((const uint8_t *)b, 256); q0.freeBuffer();
    b = q1.readBuffer(); Serial.write((const uint8_t *)b, 256); q1.freeBuffer();
    b = q2.readBuffer(); Serial.write((const uint8_t *)b, 256); q2.freeBuffer();
    b = q3.readBuffer(); Serial.write((const uint8_t *)b, 256); q3.freeBuffer();

    // slow heartbeat so you can see at a glance that it is alive
    digitalWriteFast(LED, (seq & 0x80) ? HIGH : LOW);
  }
}
