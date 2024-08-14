import glob, os
import soundfile as sf
import numpy as np
from scipy import signal
from pydub import AudioSegment
from pyannote.core import Segment, Timeline, Annotation, notebook


# prepare a 16khz mono wav file for VAD,
#  compatible with both LDC and Pyannote.
# (also fine for reaper pitch, that's not picky).
# return wav file path, not wav data.
# ¡ do not use this function for any stereo 
#	files with 1 speaker per channel.
#   in that case do VAD for each channel separately !
def wav16mono(wav_path, temp_dir = './tmp/'):
	export_path = os.path.join(temp_dir,os.path.basename(wav_path))
	if not os.path.exists(temp_dir):
		os.mkdir(temp_dir)
	wav_data = AudioSegment.from_wav(wav_path)
	wav_data = wav_data.set_channels(1)
	wav_data = wav_data.set_frame_rate(16000)
	wav_data.export(export_path, format="wav")
	return os.path.abspath(export_path)   




# read any wav and return its wav data in 16khz mono
def readwav(a_f):
	wav, sr = sf.read(a_f, dtype=np.float32)
	if len(wav.shape) == 2:
		wav = wav.mean(1)
	if sr != 16000:
		wlen = int(wav.shape[0] / sr * 16000)
		wav = signal.resample(wav, wlen)
	return wav
	
	


def fn(file_path):
	return os.path.splitext(os.path.basename(file_path))[0]



# read segments from a tsv whose first 3 columns are
# speaker_id, start_time, end_time
# return pyannote annotation
def read_segments_anno(seg_path):
	with open(seg_path, 'r') as handle:
		segments = handle.read().splitlines()
	segments = [l.split('\t') for l in segments]

	annot = Annotation()
	for l in segments:
		annot[Segment(float(l[1]),float(l[2]))] = l[0]
	return annot
	
	
	
# read a diarisation
# into list of [L]abel, [S]tart, [E]nd, [T]ranscript(initialised empty by default)
def get_transcript_segments(seg_file, tx=False):
	with open(seg_file,'r') as handle:
		f = handle.read().splitlines()
	f = [l.split('\t') for l in f]
	def _t(ln):
		if tx:
			return l[3]
		else:
			return ''
	segments = [{'l':l[0], 's':float(l[1]), 'e':float(l[2]), 't': _t(l)} for l in f]
	return segments
	
	

# read ldcbpsad output file to unlabelled pyannote Timeline
def read_ldc(ldc_path):
	timeline = Timeline()
	with open(ldc_path, 'r') as handle:
		ldc = handle.read().splitlines()
	ldc = [l.split('\t') for l in ldc]
	ldc = [l for l in ldc if 'non-speech' not in l]
	for s,e,_ in ldc:
		timeline.add(Segment(float(s),float(e)))
	return timeline
	

# save LDC (unlabelled) segmentation for Elan
# ldc_tl is a pyannote Timeline object
def save_ldc_eln(ldc_tl,save_path):
	eln = [f'LDC\t{segment.start}\t{segment.end}\t' for segment in ldc_tl]
	eln = '\n'.join(eln)
	with open(save_path,'w') as handle:
		handle.write(eln)

