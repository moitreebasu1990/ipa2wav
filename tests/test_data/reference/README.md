# Reference Audio Files

This directory contains reference audio files used for evaluating the IPA2WAV synthesis system. These files serve as the ground truth for comparing synthesized speech quality.

## File Requirements

- **Format**: WAV files (mono)
- **Sampling Rate**: 22050 Hz
- **Bit Depth**: 16-bit
- **Naming Convention**: Must match the IDs in `test_sets.json`
  - `basic_test.wav`
  - `complex_test.wav`
  - `stress_test.wav`

## Current Test Sets

1. **Basic Test** (`basic_test.wav`)
   - Contains simple phrases and words
   - Clear pronunciation
   - Neutral speaking style
   - Example phrases:
     ```
     "hello world"
     "I am a robot"
     "this is a test"
     ```

2. **Complex Test** (`complex_test.wav`)
   - Contains complex phonetic patterns
   - Multiple stress levels
   - Varied intonation
   - Example phrases:
     ```
     "international phonetic alphabet"
     "something complex with multiple stresses"
     "complicated syllables and consonant clusters"
     ```

3. **Stress Test** (`stress_test.wav`)
   - Focus on stress patterns
   - Varied emphasis levels
   - Complex word combinations
   - Example phrases:
     ```
     "stress stress stress"
     "secondary strike interrupt"
     "probably impossible understanding"
     ```

## Recording Guidelines

- Record in a quiet environment
- Use professional recording equipment if possible
- Maintain consistent volume levels
- Speak clearly and naturally
- Avoid background noise and reverb
- Leave small pauses between phrases

## Quality Check

Before using reference files, ensure:
1. Audio is clean and clear
2. No clipping or distortion
3. Correct sampling rate (22050 Hz)
4. Proper alignment with text files
5. Consistent volume levels across files