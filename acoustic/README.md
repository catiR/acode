# acode

## Acoustic features

#### Basic feature extraction

- `> pythyon3 extract_acoustic.py`

- see comments in script

- Uses audio files plus Pyannote, pydub, ldc-bpcsad, praat and/or REAPER formant tracker https://github.com/google/REAPER/ optionally findsylls.octave from https://languagelog.ldc.upenn.edu/nll/?p=46144

- Requires opensauce-python, https://github.com/voicesauce/opensauce-python 
    - available if you clone this repo with submodules, otherwise add it to `tools`
    - VoiceSauce (opensauce) requires at least one of Reaper, Praat, and Snack. `extract_acoustic.py` is configured for Reaper and/or Praat.
    - Follow opensauce readme to make sure opensauce can find reaper and praat installations. Recommended to install standalone (C) Reaper in `tools`, not use pyreaper.
    - May be temporary to be replaced by something else like parselmouth
    

#### Forced alignment to expert (human) transcriptions

- `> pythyon3 acode_align.py`

- Require ctc-forced-aligner (icelandic adaptation) https://github.com/catiR/ctc-forced-aligner
    - in development, adaptation may have bad choices need fixing

- Optionally recombine forced alignments with Pyannote and/or LDC speech activity detector for better pause detection. Also requires pydub, Pyannote.

- Output transcripts compatible with ELAN.

- Output is done and provided on project drive.


#### Derived feature calculation

- `> pythyon3 acode_acoustic.py`

- Uses previously extracted acoustic features, plus diarisations (timed Pyannote diarisation speaker labels or force alignment transcripts, tsv format compatible with ELAN).

- Produces acoustic features compatible with https://github.com/antonkarl/acode/tree/main/featureExtraction to use together as input to classification

- See comments



#### Corpus management

- find wav and transcript files on the project drive with compile_nextcloud_files() from acode_util. current as of mid january 2025.


## TODO maybe

- Completely automated pipeline - https://github.com/antonkarl/acode/tree/acousticpipeline/acoustic but preferably need Hreimur ASR
- Add speech/articulation rate feature, phonemes or syllables per second
- Make the syllable tracker do better...
- Add initial pause features, e.g. tracking Participant's pauses of at least 1 second directly following Interviewer's questions
- Add other voice/spectral features: deCODE https://github.com/cadia-lvl/deCODE/ , parselmouth ...
- Smaller timescale feature representations for classifier input - replace or supplement single value per feature per file
- Visualise and interpret data



