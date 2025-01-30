# acode

## Forced alignment to expert (human) transcriptions

- Require ctc-forced-aligner (icelandic adaptation) https://github.com/catiR/ctc-forced-aligner

- Optionally recombine forced alignments with LDC speech activity detector for better pause detection. Also requires pydub, Pyannote.

- **Run** `> pythyon3 acode_align.py`

- Output transcripts compatible with ELAN.

- Output is done and provided on project drive.



## Acoustic features

- Uses audio files plus diarised forced alignments.

- Produces acoustic features compatible with https://github.com/antonkarl/acode/tree/main/featureExtraction to use together as input to classification

- Requires: Pyannote, pydub, soundfile, plus REAPER formant tracker https://github.com/google/REAPER/ and findsylls.octave from https://languagelog.ldc.upenn.edu/nll/?p=46144

- **Run** `> pythyon3 acousticfeatures.py`

- Average duration of speech segments

- Average duration of pause segments (not including pauses before speaker begins new turn)

- Average number of pauses per minute during the speaker's turns

- Percent of time speaking (vs. pausing) during the speaker's turns, also expressed with ratio of speech to pause time during speaker's turns

- Pitch range, 10th to 90th percentile across the recording as a whole, in semitones

- Average number of words per utterance (pause group) and total average word duration, based on transcripts

- Average number of syllables per utterance (pause group) and overall average syllable duration, based on language independent estimated syllable detection

- The syllable detection can be unreliable, don't use it yet

- ⚠️🍝


#### Corpus management

- organise wav and transcript files on the project drive with compile_nextcloud_files() from acode_util





## TODO maybe

- Completely automated pipeline - https://github.com/antonkarl/acode/tree/acousticpipeline/acoustic but preferably need Hreimur ASR
- Automated no ASR version - Pyannote speaker diarisation
- Use Praat instead of REAPER for F0 tracking - if needed for closer reproduction
- Add speech/articulation rate feature, phonemes or syllables per second
- Make the syllable tracker do better...
- Add initial pause features, e.g. tracking Participant's pauses of at least 1 second directly following Interviewer's questions
- Add other voice/spectral features: deCODE https://github.com/cadia-lvl/deCODE/, VoiceSauce, ...
- Smaller timescale feature representations for classifier input - replace or supplement single value per feature per file
- Visualise and interpret data


#### Installing ldc-bpcsad

This requires HTK obtained independently. 

Here are LDC's instructions: http://linguistic-data-consortium.github.io/ldc-bpcsad/install.html

This installation also works:

`git clone https://github.com/Linguistic-Data-Consortium/ldc-bpcsad.git`

- Obtain "HTK 3.4.1 source code" (not a compiled binary), and unzip its contents to `ldc-bpcsad/tools/htk/`
    -  do not unzip as `ldc-bpcsad/tools/HTK-3.4.1/htk/`, this will not work for LDC.

```
cd ldc-bpcsad
chmod u+x tools/htk/configure
pip install wheel
```

```
sudo apt-get install gcc-multilib make patch libsndfile1
```
**specific to OS; skip unless install_htk.sh fails with relevant error message
   
```
sudo tools/install_htk.sh --njobs 1 --stage 2 PLACEHOLDER
```

**don't need to replace the placeholder with any other word. install script requires the number of arguments but this argument will never be used for anything.

`pip install . `

- Now the LDC executable is at `acodenv/bin/ldc-bpcsad`. Location can end up elsewhere depending on OS and virtual environment setup.


#### Using PyAnnote diarisation offline

- Removed, but TODO instructions for data compliance if returned.


