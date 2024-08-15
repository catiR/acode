# acode

## Acoustic features

- Requires: Pyannote, pydub, soundfile, plus REAPER formant tracker https://github.com/google/REAPER/

**Run** `> pythyon3 preprocess.py`

## Preprocessing

#### Automatic speaker diarisation + transcription pipeline

- Pyannote speech activity detection, segmentation, speaker diarisation

- Whisper ASR for diarised segments

- Second-pass speech/pause detection via Wav2Vec2 forced alignment

- Output transcripts compatible with Elan

#### Setup and requirements

- General python requirements: Pytorch (preferably with cuda), faster-whisper, transformers, pydub, pyannote.audio
- Can use virtual environment and install requirements with pip.

`python3 -m venv acodenv`

`source acodenv/bin/activate`

- Due to data protection, scripts use local (on disk) copies of Whisper, Wav2vec2, and Pyannote speech processing models, which you need to download. If you don't download them, but you edit the model paths according to default package documentation, the scripts could run but transmit sensitive data to external companies' servers. Details in script.


**Run** `> pythyon3 preprocess.py`

- In `preprocess_speech()`, the defined `original_data_dir` expects the directory structure as on NextCloud. For other setups, adjust `find_nextcloud_files()`.

- In `preprocess_speech()`, comment in/out calls to functions `run_diarisation()` and `run_asr()` to select which parts of the pipeline to run. Whisper ASR is quite slow, but setting `realign=True` does not add that much extra time.

<!--#### Diarisation

Requires both **LDC BPCSAD**, which itself depends on HTK, and **pyannote VAD + diarisation**.

Here are LDC's instructions: http://linguistic-data-consortium.github.io/ldc-bpcsad/install.html

This installation also worked:

`git clone https://github.com/Linguistic-Data-Consortium/ldc-bpcsad.git`

- From http://speech.ee.ntu.edu.tw/homework/, download "HTK 3.4.1 source code (zip)" (not one of the compiled binaries), and unzip its contents to `ldc-bpcsad/tools/htk/`
    -  do not just unzip as `ldc-bpcsad/tools/HTK-3.4.1/htk/`, this will not work for LDC.

`cd ldc-bpcsad`
`chmod u+x tools/htk/configure`
`pip install wheel`
`sudo apt-get install gcc-multilib make patch libsndfile1`
   *  specific to OS; don't do that unless you try the next step and failed with relevant error message
   
`sudo tools/install_htk.sh --njobs 1 --stage 2 PLACEHOLDER`
   *  don't need to replace the placeholder with any other word, it just needs text to fill the number of arguments. that argument will never be used for anything.

`pip install . `

* Now the LDC executable is at `acodenv/bin/ldc-bpcsad`. Location can end up elsewhere depending on OS and virtual environment setup.

- **TODO:** About pyannote models for offline use 


#### ASR

- TODO: about models for offline use.

#### Acode features

- Install REAPER, TODO change this to Praat.-->



## TODO

- Use manually corrected file exported from ELAN as input to feature calculation
- Use Praat instead of REAPER for F0 tracking
- Documentation. For now see comments in code.
- Add speech/articulation rate feature, phonemes or syllables per second
- Add initial pause features, e.g. tracking Participant's pauses of at least 1 second directly following Interviewer's questions
- Add other voice/spectral features
- Continuous feature representations, replace or supplement single average value per file


