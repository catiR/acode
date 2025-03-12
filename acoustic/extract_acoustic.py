import os, subprocess, shutil, glob, sys
import numpy as np
from acode_util import *

# for pyannote only
from pyannote.audio import Pipeline
from pyannote.audio.pipelines.utils.hook import ProgressHook
from pyannote.core import Segment, Timeline, Annotation, notebook
import torch, torchaudio

# for voicesauce only
# https://github.com/voicesauce/opensauce-python
opensauce_path = os.path.join('tools','opensauce-python')
sys.path.append(opensauce_path)
from opensauce.__main__ import CLI



# - - - - - - - - - - - - - - - - - - - -
#      syllables
# - - - - - - - - - - - - - - - - - - - -

# do a syllable detection process
# if it doesnt find enough syllables,
# can try lowering MinPeak in findsylls.octave
#  (default MinPeak = 0.04)
def detect_sylls(wav_path, save_path, script_path):
	if not os.path.exists(save_path):
		input_arg = os.path.splitext(wav_path)[0]
		p = subprocess.run(['octave', script_path, input_arg]).stdout
		syll_output = input_arg + '.sylls'
		shutil.move(syll_output, save_path)
	return save_path



# - - - - - - - - - - - - - - - - - - - -
#      pitch
# - - - - - - - - - - - - - - - - - - - -

# nevler 2017:
# "collect_pitch_data_from_files.praat"
# "limits for pitch tracking were set at 75–300 Hz."
# unsure if ac or cc algorithm & several other parameters --
# collect_pitch.praat script default is 10ms step,
# other best guess defaults from praat software are - 
#  10ms step 40ms window for autocorrelation
#  0.25/pitch_floor step, 1/pitch_floor window for crosscorrelation
#   (however voicesauce requires integers)
# + various other parameters where voicesauce defaults == praat defaults
#   - consider reduce voicing threshold, 
#     increase octave cost, octave jump cost, voiced/unvoiced cost,
#     or kill octave jump switch.
def apply_voicesauce_praat(wav_file, save_file, algorithm = "ac"):

	assert algorithm in ['ac','cc'], f'Praat pitch "{algorithm}" doesnt exist'
	
	# specify praat algorithm in output
	save_dir, save_f = os.path.split(save_file)
	save_dir = os.path.join(save_dir,'praat',algorithm)
	save_file = os.path.join(save_dir, save_f)
	mds(save_file)
	
	if os.path.exists(save_file): #skip
		return save_file

	# parameters
	pitch_floor = 75
	pitch_ceil = 300
	
	if algorithm == 'ac':
		frame_shift = int(0.75/pitch_floor*1000)
		window_size = int(3/pitch_floor*1000)
		
	elif algorithm == 'cc':
		frame_shift = int(0.25/pitch_floor*1000)
		window_size = int(1/pitch_floor*1000)

	vs_args = ['--measurements', 'praatF0', 
				#'--include-f0-column', 
				'--no-textgrid', '--no-labels',
				'--f0', 'praatF0', 
				'--frame-shift', str(frame_shift),
				'--window-size', str(window_size),
				'--praat-f0-method', algorithm,
				'--praat-min-f0', str(pitch_floor),
				'--praat-max-f0', str(pitch_ceil),
				'--output-settings-path', os.path.join(save_dir,'.settings'),
				'-o', save_file, 
				wav_file]
	
	try:
		vs_cli = CLI(args = vs_args)
		vs_cli.process()
	except (OSError, IOError, ValueError) as err:
		print(err)

	return save_file


# ¡ not implementing for now
#   because voicesauce is problems ¡
#def apply_voicesauce_reaper(wav_file, save_file, reaper_path):
#	vs_args = ['--measurements', 'reaperF0', 
#				'--f0', 'reaperF0', 
#				'--reaper-path', reaper_path, 
#				'-o', save_file, 
#				wav_file]
#	try:
#		vs_cli = CLI(args = vs_args)
#		vs_cli.process()
#	except (OSError, IOError, ValueError) as err:
#		print(err)
#	return save_file



# reaper output f0 data as list of Time, F0 if exists, voicing indicator
# -- neveler: 
#   estimator bounds are set following
#    https://www.ling.upenn.edu/courses/Spring_2018/ling620/Nevler2017.pdf
#   - added fallback to reaper standard estimation range if 
#     percent of voiced frames in file is less than min_voicing,
#     set min_voicing=0 to disable this
# -- hirst: 2-pass estimation 
# 	  " The analysis by synthesis of speech melody: from data to models "
#   originally for praat tracker but since used on others
def apply_reaper(wav_path, save_path, reaper_path, method = 'nevler', min_voicing=0.3):


	def _reaper(reaper_path, wav_path, maxf0, minf0):
		f0_raw_data = subprocess.run([reaper_path, "-i", wav_path, 
			'-f', '/dev/stdout', '-x', maxf0, '-m', minf0, '-a'],
			capture_output=True).stdout
		f0_raw_data = f0_raw_data.decode()
		return f0_raw_data
	
	method = method.lower()
	assert method in ('hirst','nevler'),  f'Reaper strategy "{method}" doesnt exist'
	
	save_dir, save_f = os.path.split(save_path)
	save_dir = os.path.join(save_dir,method)
	save_path = os.path.join(save_dir, save_f)
	mds(save_path)
	
	if os.path.exists(save_path): # skip
		return save_path

	if method == 'nevler':
		max_f0='300'
		min_f0='75'
		
		f0_raw_data = _reaper(reaper_path, wav_path, max_f0, min_f0)
		f0_data = get_reaper(f0_raw_data)
	
		if sum([v for t,f,v in f0_data]) < (len(f0_data) * min_voicing):
			print(' F0 tracking failed, retrying with expanded search range')
			f0_raw_data = _reaper(reaper_path, wav_path, '500', '40')
			f0_data =  get_reaper(f0_raw_data)
	

	elif method == 'hirst':
		max_f0='700'
		min_f0='50'
		
		f0_1pass_rd = _reaper(reaper_path, wav_path, max_f0, min_f0)
		f0_1pass = get_reaper(f0_1pass_rd)
		
		f0_1pass = [f for t,f,v in f0_1pass if v==1]
		q1 = np.quantile(f0_1pass,0.25)
		q3 = np.quantile(f0_1pass,0.75)
		
		pceil = 2.5 * q3
		# originally was 1.5 not 2.5, reduce if 2.5 produces octave doubling
		# hirst says that's rare in practise and recommends 2.5 
		#  as 1.5 can be below speaker's actual pitch ceiling.
		# bounds ( .75*q1, 2.5*q3 ) produce good percentage of voiced frames for ACODE.
		# if ( .75*q1, 1.5*q3 ) have low percentage voiced speech,
		#   could retry with expansion to 2.5*q3 and/or .5*q1
		
		pfloor = 0.75 * q1
		max_f0 = str(round(pceil))
		min_f0 = str(round(pfloor))
		
		f0_raw_data = _reaper(reaper_path, wav_path, max_f0, min_f0)
		f0_data = get_reaper(f0_raw_data)
		

	final_voiced = sum([v for t,f,v in f0_data])/len(f0_data)
	print(f' * [R-{method}] voiced frames: {final_voiced:.2f}')
	if final_voiced < min_voicing:
		print('\tNOTICE: Poor detection of voiced speech ^')
	
	if not os.path.exists(save_path):
		with open(save_path, 'w') as handle:
			handle.write(f0_raw_data)

	return save_path




# - - - - - - - - - - - - - - - - - - - -
#      vad, diarisation
# - - - - - - - - - - - - - - - - - - - -


# run ldc sad process
# 
# minimum duration for speech segments: 250ms
# minimum duration for silent pauses: 150ms
#  following cho et al 2021, Lexical and Acoustic 
#  Characteristics of Young and Older Healthy Adults
def ldc_sad(wav_path, sad_cmd, lab_path):
    
    if not(os.path.exists(lab_path)):
        sad_proc = subprocess.call([sad_cmd, "--channel", "1", 
        "--output-dir", os.path.dirname(lab_path), "--speech", "0.250", 
        "--nonspeech", "0.150", wav_path])
        
    return lab_path
#
#### Installing ldc-bpcsad
#	This requires HTK obtained independently. 
#	Here are LDC's instructions: 
#	http://linguistic-data-consortium.github.io/ldc-bpcsad/install.html
#
# This installation also works:
#	`git clone https://github.com/Linguistic-Data-Consortium/ldc-bpcsad.git`
# - Obtain "HTK 3.4.1 source code" (not a compiled binary), and unzip its contents to `ldc-bpcsad/tools/htk/`
# - do not unzip as `ldc-bpcsad/tools/HTK-3.4.1/htk/`, this will not work for LDC.
# 	```
# 	cd ldc-bpcsad
# 	chmod u+x tools/htk/configure
# 	pip install wheel
# 	```
#	sudo apt-get install gcc-multilib make patch libsndfile1
#**specific to OS; skip unless install_htk.sh fails with relevant error message
#
#	sudo tools/install_htk.sh --njobs 1 --stage 2 PLACEHOLDER
#**don't need to replace the placeholder with any other word. install script requires the number of arguments but this argument will never be used for anything.
#
#	pip install .
#
# - Now the LDC executable is at `yourvenv/bin/ldc-bpcsad` or on path in venv.
#    Location can end up elsewhere depending on OS and virtual environment setup.
    


# initialise Pyannote VAD + speaker diarisation
#
# to use pyannote vad locally without access token 
#  for private local processing of sensitive data,
# download the pytorch model.bin and config.yaml 
#   from https://huggingface.co/pyannote/wespeaker-voxceleb-resnet34-LM/tree/main
#   also the model.bin and config.yaml from https://huggingface.co/pyannote/segmentation-3.0/tree/main
# and download config.yaml and handler.py
#   from https://huggingface.co/pyannote/speaker-diarization-3.1/tree/main
# then edit the embedding and segmentation in the pyvad config file defined above
#   to point at your local wespeaker embedding & segmentation3.0 model.bin files
# ex. <embedding: tools/pyannote-speakerdia-3.1/wespeaker-voxceleb-resnet34-LMpytorch_model.bin>
#	 <segmentation: tools/pyannote-speakerdia-3.1/segmentation30pytorch_model.bin>

def setup_pya_vad(vad_yaml_path):
	vad = Pipeline.from_pretrained(vad_yaml_path)
	if torch.cuda.is_available():
		vad.to(torch.device("cuda"))
	
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

def get_pya_vad(wav,save_path,vad_pipeline):

	speaker_labels = {"main": "V", "interviewer": "S", "unknown": "X"}
	
	if os.path.exists(save_path):
		print(f'Pyannote already exists for {fn(wav)}, skipping')
		return save_path, speaker_labels

	else:
		wf, sr = torchaudio.load(wav) # supposedly faster
		with ProgressHook() as hook:
			pya_vad = vad_pipeline({"waveform": wf, "sample_rate": sr},
				 num_speakers=2, hook=hook)
			
		main_speaker = pya_vad.chart()[0][0]
		other_speaker = pya_vad.chart()[1][0]
		with open(save_path,'w') as handle:
			for segment,track,label in pya_vad.itertracks(yield_label=True):
				if label == main_speaker:
					output_label = speaker_labels['main']
				elif label == other_speaker:
					output_label = speaker_labels['interviewer']
				else:
					print("WARNING! this shouldn't be used",
							" to diarise files with more than 2 speakers")
					output_label = label
				handle.write(f'{output_label}\t{segment.start}\t{segment.end}\n')
	return save_path, speaker_labels
	
	
	

# heuristic transfer 2-speaker diarisation labels
#   from broadly segmented pyannote diarisation
#   onto sensitive LDC segmentation that doesnt have any speaker labels.
# automatic speaker labelling assumes the patient/control speaks most
def transfer_2spk_labels(pya_file,ldc_file,save_file,speaker_labels):

	# read Pya diarisation to Annotation
	pya_dia = read_diarisation(pya_file)

	# ldc lab file has no speaker labels, read to Timeline
	ldc_sad = Timeline()
	with open(ldc_file, 'r') as handle:
		ldc_data = handle.read().splitlines()
	ldc_data = [l.split('\t') for l in ldc_data]
	ldc_data = [l for l in ldc_data if 'non-speech' not in l]
	for s,e,_ in ldc_data:
		ldc_sad.add(Segment(float(s),float(e)))
	

	# assume experiment participant is 
	#   the speaker detected as talking the most
	# this can be wrong!
	# which is very bad for all the analysis later!
	main_speaker = pya_dia.chart()[0][0]
	other_speaker = pya_dia.chart()[1][0]
    
	transferred = Annotation()
	for lseg in ldc_sad:

		# find all labelled PYA segments that overlap with the LDC segment
		candidates = [(pseg, label) 
						for pseg,track,label in pya_dia.itertracks(yield_label=True)
						if lseg & pseg]
		if not candidates:
		# if no pyannote label, assign the segment to speaker X
		#	print(f'NOTICE: PyAnnote did not label speech for LDC segment {lseg}')
			transferred[lseg] = speaker_labels['unknown']
		else:
		# pyannote segmentation has overlapping segments and ldc doesn't
		# so if both people speak at some point during this segment,
		# assign it to the participant for now
			candidates = [spk for seg,spk in candidates]
			if main_speaker in candidates:
				transferred[lseg] = speaker_labels['main']
			else:
				transferred[lseg] = speaker_labels['interviewer']

	# save new diarisation
	pya2eln(transferred,save_file)
	return save_file
    
	

# - - - - - - - - - - - - - - - - - - - -
#      run audio processing on corpus
# - - - - - - - - - - - - - - - - - - - -

# inputs: dict about how to find files on disk,
#  and dict of external feature extractor paths
def featurise_speech(speech_corpus, extractors, save_dir):

	reaper_path = extractors['reaper_path']
	syll_detector = extractors['syll_detector']
	sad_executable = extractors['sad_executable']
	pyannote_config = extractors['pyannote_config']
		
	temp_wav_dir = os.path.join(save_dir, 'tmp')
	feats_dir = os.path.join(save_dir, 'feats')
	
	vad = setup_pya_vad(pyannote_config)

	for fid, finfo in speech_corpus.items():
	
		print(f'Acoustic features in audio {fid} ...')
		
		wav_orig = finfo[0]
		wav_tmp = wav16mono(wav_orig, os.path.join(temp_wav_dir, f'{fid}.wav'))
		
		#voicesauce_outs = os.path.join(feats_dir,f'voicesauce/{fid}.tsv')
		reaper_outs = os.path.join(feats_dir,f'reaper/{fid}.f0')
		syll_file = os.path.join(feats_dir,f'sylls/{fid}.sylls')
		mds(syll_file)
		
		#_ = apply_voicesauce_praat(wav_tmp, voicesauce_outs)
		_ = apply_reaper(wav_tmp, reaper_outs, reaper_path)
		_ = apply_reaper(wav_tmp, reaper_outs, reaper_path, method= "hirst")
		_ = detect_sylls(wav_tmp, syll_file, syll_detector)
		
		# pyannote automatic diarisation
		pya_file = os.path.join(save_dir,'diarised','pya',f'{fid}.txt')
		mds(pya_file)
		_, speaker_labels = get_pya_vad(wav_tmp,pya_file,vad)
		
		# ldc speech activity detection
		ldc_file = os.path.join(feats_dir,'lab',f'{fid}.lab')
		mds(ldc_file)
		_ = ldc_sad(wav_tmp, sad_executable, ldc_file)
		
		# label ldc speech segments with pyannote speaker labels
		pyaldc_file = os.path.join(save_dir,'diarised','pya_ldc',f'{fid}.txt')
		mds(pyaldc_file)
		_ = transfer_2spk_labels(pya_file,ldc_file,pyaldc_file,speaker_labels)



if __name__ == "__main__":

	audio_corpus_dir = '/home/cati/proj/acode/NextCloud/Data/'
	features_save_dir = '../../acoustic-processing/output-is020/'
	
	
	# path to REAPER F0 extractor executable
	# get it & build from:
	# https://github.com/google/REAPER/
	# 
	# syll_detector
	# syllable detection from http://languagelog.ldc.upenn.edu/myl/findsylls
	#
	# pyannote_config 
	# path to yaml file, see comments above
	#
	# ldc-bpcsad executable path
	# see comments above
	
	extractors = {
		'reaper_path': './tools/REAPER/build/reaper',
		'syll_detector' : './tools/findsylls.octave',
		'sad_executable' : 'ldc-bpcsad',
		'pyannote_config' : "./tools/pyannote-speakerdia-3.1/diarization31config.yaml"
		}

	data_files = compile_nextcloud_files(audio_corpus_dir, keep_all_audios=True)
	
	featurise_speech(data_files, extractors, features_save_dir)
	
	
 
	
