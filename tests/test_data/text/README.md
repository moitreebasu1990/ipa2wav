# Test Text Files

This directory contains IPA (International Phonetic Alphabet) text files used for evaluating the IPA2WAV synthesis system. These files provide the input text for speech synthesis testing.

## File Requirements

- **Format**: Plain text files (`.txt`)
- **Encoding**: UTF-8
- **Line Endings**: Unix-style (`\n`)
- **Naming Convention**: Must match the IDs in `test_sets.json`
  - `basic_test.txt`
  - `complex_test.txt`
  - `stress_test.txt`

## Current Test Sets

1. **Basic Test** (`basic_test.txt`)
   - Simple IPA sequences
   - Basic words and phrases
   - Current content:
     ```
     həˈloʊ wɜrld
     aɪ æm ə ˈroʊbɑt
     ðɪs ɪz ə ˈtɛst
     ```

2. **Complex Test** (`complex_test.txt`)
   - Complex phonetic patterns
   - Multiple stress markers
   - Current content:
     ```
     ˌɪntərˈnæʃənəl ˌfəˈnɛtɪk ˈælfəˌbɛt
     ˈsʌmθɪŋ kəmˈplɛks wɪð ˈmʌltɪpəl ˈstrɛsɪz
     ˈkɑmplɪˌkeɪtɪd ˈsɪləbəlz ænd ˈkɑnsənənt ˈklʌstərz
     ```

3. **Stress Test** (`stress_test.txt`)
   - Focus on stress patterns
   - Various stress combinations
   - Current content:
     ```
     ˈstrɛs ˌstrɛs ˈstrɛs
     ˌsɛkənˈdɛri ˈstraɪk ˌɪntəˈrʌpt
     ˈprɑbəbli ˌɪmpɑˈsɪbəl ˌʌndərˈstændɪŋ
     ```

## IPA Symbols Used

- **Vowels**: æ, ɑ, ə, ɛ, ɪ, i, oʊ, ʌ, ɜ
- **Consonants**: b, d, f, g, h, k, l, m, n, p, r, s, t, v, w, z, θ, ð, ʃ, ŋ
- **Stress Markers**: 
  - `ˈ` Primary stress
  - `ˌ` Secondary stress

## File Format Guidelines

1. **One utterance per line**
2. **Proper spacing**:
   - Space between words
   - No trailing/leading spaces
   - No empty lines between utterances
3. **Correct stress marking**:
   - Primary stress before stressed syllable
   - Secondary stress where appropriate
4. **Consistent transcription**:
   - Use standard IPA symbols
   - Maintain consistent style across files

## Quality Check

Before using test files, ensure:
1. All IPA symbols are valid
2. Stress markers are properly placed
3. Files are properly encoded (UTF-8)
4. No missing or extra spaces
5. Transcriptions match reference audio