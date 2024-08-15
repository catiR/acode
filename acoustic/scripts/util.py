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
	


#filename
def fn(file_path):
	return os.path.splitext(os.path.basename(file_path))[0]



# organise paths to original recordings + transcripts, 
# and output of diarisations and ASR,
# assuming they(/will) exist in a certain directory structure
# TODO: path format not OS dependent
def find_nextcloud_files(recording_dir = '../../Data/', output_dir="./acoustic-output/"):
	control_prefix = os.path.join(recording_dir,'Controls/Audio/')
	patient_prefix = os.path.join(recording_dir,'Patients/AUDIO/')
	
	control_wavs = glob.glob(control_prefix+'*/*.wav')
	patient_wavs = glob.glob(patient_prefix+'*/*.wav')

	
	def _fd(file_path, prefix_dir):
		d = os.path.relpath(os.path.dirname(file_path), prefix_dir)
		d = d.replace('audio','')
		return(os.path.join(d,fn(file_path)))
		
	try:
		assert len(set([fn(f) for f in control_wavs])) == len(control_wavs)
	except:
		raise Exception("Wav files didn't have unique names. Check for duplicates, rename them if not duplicate.")

	
	# wav: audio file
	# pya: pyannote diarisation
	cpx,ppx = control_prefix, patient_prefix
	files_dict = {'control' : {fn(f) : 
			{'wav' : f, 
			 'pya': f'{output_dir}Controls/diarisation/{_fd(f,cpx)}-PyaVad.txt',
			 'lab': f'{output_dir}Controls/diarisation/{_fd(f,cpx)}.lab',
			 'ldc': f'{output_dir}Controls/diarisation/{_fd(f,cpx)}-LDC.txt',
			 'f0': f'{output_dir}Controls/features/{_fd(f,cpx)}.f0',
			 'asrP': f'{output_dir}Controls/asr/{_fd(f,cpx)}-PyaVad-' # prefix for asr paths
			 } 
			for f in control_wavs },
			'patient' : {fn(f) : 
			{'wav' : f, 
			 'pya': f'{output_dir}Patients/diarisation/{_fd(f,ppx)}-PyaVad.txt',
			 'lab': f'{output_dir}Patients/diarisation/{_fd(f,ppx)}.lab',
			 'ldc': f'{output_dir}Patients/diarisation/{_fd(f,ppx)}-LDC.txt',
			 'f0': f'{output_dir}Patients/features/{_fd(f,ppx)}.f0',
			 'asrP': f'{output_dir}Patients/asr/{_fd(f,ppx)}-PyaVad-'
			 }
			for f in patient_wavs } }
	
	for v in list(files_dict['control'].values()) + list(files_dict['patient'].values()):
		dia_d = os.path.dirname(v['pya'])
		feat_d = os.path.dirname(v['f0'])
		asr1_d = os.path.dirname(v['asrP']).replace('/asr/', '/asr-1pass/')
		as2p_d = os.path.dirname(v['asrP']).replace('/asr/', '/asr-2pass/')
		for d in [dia_d, feat_d, asr1_d, as2p_d]:
			if not os.path.exists(d):
				print('making', d)
				os.makedirs(d)

			
	return files_dict 



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

