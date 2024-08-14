from scripts.util import *
import scripts.asrs as asr
import os, subprocess, glob
from pyannote.audio import Pipeline
from pyannote.audio.pipelines.utils.hook import ProgressHook
from pyannote.core import Segment, Timeline, Annotation, notebook
import torchaudio
import numpy as np
from scripts.ctcalign import aligner




# organise paths to original recordings + transcripts, 
# and output of diarisations and ASR,
# assuming they(/will) exist in a certain directory structure
# TODO: path format not OS dependent
def find_nextcloud_files(recording_dir = '../../Data/', output_dir="./output/"):
	control_prefix = os.path.join(recording_dir,'Controls/')
	patient_prefix = os.path.join(recording_dir,'Patients/')
	
	control_wavs = glob.glob(control_prefix+'A*/*/*.wav')
	patient_wavs = glob.glob(patient_prefix+'A*/*/*.wav')

	
	for fdir in [output_dir, f'{output_dir}diarisation',f'{output_dir}asr']:
		if not os.path.exists(fdir):
			os.mkdir(fdir)
	
	
	# wav: audio file
	# pya: pyannote diarisation
	files_dict = {'control' : {fn(f) : 
			{'wav' : f, 
			 'pya': f'{output_dir}diarisation/{fn(f)}-PyaVad.txt',
			 'lab': f'{output_dir}diarisation/{fn(f)}.lab',
			 'ldc': f'{output_dir}diarisation/{fn(f)}-LDC.txt',
			 'asrP': f'{output_dir}asr/{fn(f)}-PyaVad-' # prefix for asr paths
			 } 
			for f in control_wavs },
		'patient' : {fn(f) : 
			{'wav' : f, 
			 'pya': f'{output_dir}diarisation/{fn(f)}-PyaVad.txt',
			 'lab': f'{output_dir}diarisation/{fn(f)}.lab',
			 'ldc': f'{output_dir}diarisation/{fn(f)}-LDC.txt',
			 'asrP': f'{output_dir}asr/{fn(f)}-PyaVad-'
			 }
			for f in patient_wavs } }
			
	return files_dict 
	
	
	
	
# initialise pyannote
def setup_pya_vad(vad_yaml_path):
	vad = Pipeline.from_pretrained(vad_yaml_path)
	
	# min_duration_on is min speech duration
	# should be 250ms to match cho et al 2021,
	#   but this parameter can't actually be set in pyannote 3.1
	# min_duration_off is min pause duration, should be 150ms
	HYPER_PARAMETERS = { 'segmentation' :
	{#"min_duration_on": 0.25,
	"min_duration_off": 0.15}
	}
	
	# pyannote 3.1 has default minimum pause duration 0.0 not 0.15, 
	#  but even pyannote with 0.0 finds less short pauses than LDC does with 0.15.
	# therefore, leaving hyperparameter instantiation commented out to run with default.
	#vad.instantiate(HYPER_PARAMETERS)
	#pth = vad.parameters(instantiated=True)
	#print(pth)
	return vad
	


# run pyannote on a file
#  or load its output from file if it already existed
def get_pya_vad(wav,save_path,labels_dict,vad_pipeline):
	if os.path.exists(save_path):
		print(f'Skipping {fn(wav)}, pyannote already exists')
		with open(save_path,'r') as handle:
			pya_data = handle.read().splitlines()
		pya_vad = Annotation()
		pya_data = [l.split('\t') for l in pya_data]
		for l,s,e in pya_data:
			pya_vad[Segment(float(s),float(e))]=l

	else:
		wf, sr = torchaudio.load(wav) # supposedly faster
		with ProgressHook() as hook:
			pya_vad = vad_pipeline({"waveform": wf, "sample_rate": sr}, num_speakers=2, hook=hook)
			
		main_speaker = pya_vad.chart()[0][0]
		other_speaker = pya_vad.chart()[1][0]
		with open(save_path,'w') as handle:
			for segment,track,label in pya_vad.itertracks(yield_label=True):
				if label == main_speaker:
					output_label = labels_dict['main']
				elif label == other_speaker:
					output_label = labels_dict['interviewer']
				else:
					print("WARNING! edit the program before using it on files with more than 2 speakers")
					output_label = label
				handle.write(f'{output_label}\t{segment.start}\t{segment.end}\n')
	return pya_vad

 


# to use pyannote vad locally without access token,
# download the pytorch model.bin and config.yaml 
#   from https://huggingface.co/pyannote/wespeaker-voxceleb-resnet34-LM/tree/main
#   also the model.bin and config.yaml from https://huggingface.co/pyannote/segmentation-3.0/tree/main
# and download config.yaml and handler.py
#   from https://huggingface.co/pyannote/speaker-diarization-3.1/tree/main
# then edit the embedding and segmentation in the pyvad config file defined above
#   to point at your local wespeaker embedding & segmentation3.0 model.bin files
# ex. <embedding: localmodels/pyannote-speakerdia-3.1/wespeaker-voxceleb-resnet34-LMpytorch_model.bin>
#	 <segmentation: localmodels/pyannote-speakerdia-3.1/segmentation30pytorch_model.bin>
def run_pya_diarisations(data_files):

	# decide how to label speakers in output
	labels_dict = {"main": "V", "interviewer": "S", "unknown": "XX"}
	
	vad = setup_pya_vad("localmodels/pyannote-speakerdia-3.1/diarization31config.yaml")
	

	for participant_group in ['control', 'patient']:
		for speaker,file_paths in data_files[participant_group].items():
			wav = file_paths['wav']
			print(f'Diarising sample {fn(wav)}')
			tmp_wav_path = wav16mono(wav)
			pya_vad = get_pya_vad(tmp_wav_path,file_paths['pya'],labels_dict,vad)
		




def run_asr(data_files,asr_method, realign=False):

	# every named model needs a path to the model on disk,
	# and the name of the function in asrs.py that processes its architecture.
	# use local models for data protection i.e. don't send clinical speech to OpenAI servers
	asr_models = {'whisper': 
			{'path': './localmodels/LVL/whisper-large-icelandic-62640-steps-967h-ct2',
			'recogniser':'recognise_fasterwhisper'}, 
		'wav2vec2': 
			{'path': './localmodels/LVL/wav2vec2-large-xlsr-53-icelandic-ep30-967h', 
			'recogniser': 'recognise_w2v2'}}
	# TODO:
	# user define these paths somewhere better
			
		
	try:
		asr_model_path = asr_models[asr_method]
	except:
		print(f"Didn't find a model for ASR name {asr_method}. ",sep='')
		print(f"Available models are {' '.join(list(asr_models.keys()))}, ",sep='')
		print(f"or add more asr_models in run_asr().")



	if realign:
		f_model_path = asr_models['wav2vec2']['path']
		f_model_word_sep = '|'
		f_model_blank_tk = '[PAD]'
		w2v2_aligner = aligner(f_model_path,f_model_word_sep,f_model_blank_tk)
		

	for participant_group in ['control', 'patient']:
		for speaker,file_paths in data_files[participant_group].items():
			print(f'ASR for {speaker}: ... {asr_method}')
			wav = file_paths['wav']
			segs = file_paths['pya']
			asr_prefix = file_paths['asrP']
			asr_file = asr.asr_one(wav,segs,asr_prefix,asr_method,asr_models[asr_method])
			
			
			if realign:
				_ = asr.realign_w2v2(wav,asr_file,w2v2_aligner)




# refer to comments in each pipeline function
def preprocess_speech():
	original_data_dir = '/home/cati/proj/acode/NextCloud/Data/'
	save_dir = './output/'
	
	asr_method = 'whisper'
	
	data_files = find_nextcloud_files(original_data_dir, save_dir)
	
	
	run_pya_diarisations(data_files)
	run_asr(data_files,asr_method, realign=True)


	
if __name__ == "__main__":
	preprocess_speech()
	
   
