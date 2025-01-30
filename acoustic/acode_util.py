import os, glob, re
from pydub import AudioSegment
from pyannote.core import Segment, Timeline, Annotation



# - - - - - audio - - - - -


# prepare a 16khz mono wav file
# return wav file path, not wav data.
# ¡ do not use this function for any stereo 
#	files with 1 speaker per channel.
#   in that case process each channel separately !
def wav16mono(wav_path, export_path):
	if not os.path.exists(export_path):
		wav_data = AudioSegment.from_wav(wav_path)
		wav_data = wav_data.set_channels(1)
		wav_data = wav_data.set_frame_rate(16000)
		wav_data.export(export_path, format="wav")
	return os.path.abspath(export_path)



# - - - - - nextcloud files - - - - -


#filename
def fn(file_path):
	return os.path.splitext(os.path.basename(file_path))[0]


#participant type
def pargroup(pgroup,wavpath):
	cohort = re.search('-cohort\d\([SM]C?[DI]\)', wavpath)
	if cohort:
		pgroup = f'{pgroup}{cohort.group()}'
	return pgroup
	

# get original files from paths on the specific nextcloud drive
# ret. 'fileid' : ['AudioFile.wav', 'TranscriptFile.txt', 'ParticipantGroup']
# if keep_all_audios True, use audio even does not have a gold transcript
def compile_nextcloud_files(
		nextcloud_dir='../NextCloud/Data/',
		keep_all_audios=False,
		):

	control_prefix = os.path.join(nextcloud_dir,'Controls/Audio/')
	patient_prefix = os.path.join(nextcloud_dir,'Patients/AUDIO/')

	control_wavs = glob.glob(control_prefix+'*/*.wav')
	patient_wavs = glob.glob(patient_prefix+'*/*/*.wav')
		
	try:
		assert len(set([fn(f) for f in control_wavs])) == len(control_wavs)
		assert len(set([fn(f) for f in patient_wavs])) == len(patient_wavs)
	except:
		raise Exception(("Wav files didn't have unique names. " 
		"Check for duplicates, rename them if not duplicate."))

	# get transcripts -
	# some control files are nonstandard
	nextcloud_files = {}
	for wf in control_wavs:
		xf = wf.replace('/Audio/','/Transcriptions/')
		xf = xf.replace('-audio(CONTROLS)','-transcripts(CONTROL)')
		xf = xf.replace('Childhood Home(CH)-','Childhood Home (CH)-')
		xf = xf.replace('.wav','-trans.txt')
		xf = xf.replace('TP_2C-KVK-1955-1-trans.txt',
							'TP_2C-KVK-1955-1 -trans.txt')
		if not os.path.exists(xf):
			xf = (xf[::-1].replace('C_K'[::-1], 'C-K'[::-1], 1))[::-1]
		if not os.path.exists(xf):
			xf = False
		if xf or keep_all_audios:
			nextcloud_files[fn(wf)] = [wf,xf or 'NOT TRANSCRIBED','control']
		
	# several patient files are nonstandard
	# current to 19.01.2025
	for wf in patient_wavs:
		xf = wf.replace('/AUDIO/','/TRANSCRIPTIONS/')
		xf = xf.replace('-audio(PATIENTS)','-transcripts(PATIENTS)')
		xf = xf.replace('Home(CH)-transcripts(PATIENTS)',
							'Home(CH)-transcriptions (PATIENTS)')
		xf = xf.replace('.wav','.txt')
		if not os.path.exists(xf):
			xf = xf.replace('.txt', '-trans.txt')
			xf = xf.replace('011(no detour)-trans.txt','011-trans(no detour).txt')
		if not os.path.exists(xf):
			xf = xf.replace('_P-ACODE-005-trans.txt','_P-ACODE-005.txt')
			xf = xf.replace('ACODE-017-trans.','ACODE-017trans.')
			xf = xf.replace('039(3NoDetour','039(3)')
			xf = xf.replace('CH_P-ACODE-052(3)NoDetour-trans',
								'CH_P-ACODE-052(3)NoDetour')
			xf = xf.replace('CH_P-ACODE-043(3)NoDetour-trans',
								'CH_P-ACODE-052(3)NoDetour')
			xf = xf.replace('CH_P-ACODE-050(1)NoDetour-trans',
								'CH_P-ACODE-052(3)NoDetour')
			xf = xf.replace('CH_P-ACODE-047(2)NoDetour-trans',
								'CH_P-ACODE-052(3)NoDetour')
		if not os.path.exists(xf):
			xf = xf.replace('NoDetour','').replace('phonecallcutout','')
		if not os.path.exists(xf):
			xfa = re.sub(r'\(\d\)-trans', '-trans', xf)
			xfb = re.sub(r'-trans.txt', '.txt', xf)
			if os.path.exists(xfa):
				xf = xfa
			if os.path.exists(xfb):
				xf = xfb
		if not os.path.exists(xf):
			xf = re.sub(r'\(\d\)-trans', '', xf)

		if not os.path.exists(xf):
			xf = False
			
		if xf or keep_all_audios:
			nextcloud_files[fn(wf)] = [wf,xf or 'NOT TRANSCRIBED','patient']
	
	return nextcloud_files



# keep track of where to throw too many files i guess
def setup_nextcloud_outputs(recording_dir = '../NextCloud/Data/', output_dir="./output/"):

	nextcloud_files= compile_nextcloud_files(recording_dir)
	
	files_dict = {'control':{},'patient':{}}
	for fid, finfo in nextcloud_files.items():
	
		wav = finfo[0]
		group = finfo[2]
		group = pargroup(group,wav)
		
		files_dict[finfo[2]][fid] = {
		'wav': wav,
		'xcp-gold': finfo[1],
		'group': group,

		# normalise txt of xcp-gold for ctc-forced-aligner
		'gold-cfanorm': os.path.join(output_dir,'align/',f'{fid}.cfanorm'),
		
		# temp wav if path to 16khz mono file is required input
		'tmp-wav': os.path.join(output_dir,'tmp/',f'{fid}.wav'),
		
		# lab file output from ldc-bpcsad
		'ldc-sad-lab': os.path.join(output_dir,'feats/',f'{fid}.lab'),
		
		# reaper formant tracks
		'f0': os.path.join(output_dir,'feats/',f'{fid}.f0'),
		
		# syllable points
		'syl': os.path.join(output_dir,'feats/', f'{fid}.sylls'),
		
		# json outputs from ctc-forced-aligner
		'gold-cfalnj': os.path.join(output_dir,'align/',f'{fid}.json'),
		
		# assign times from gold-cfa-align-json back to words in original transcript
		'gold-cfaspk': os.path.join(output_dir,'diarised/', f'{fid}-cfa.txt'),
		
		# combine cfa's timed diarised words with ldc-bpcsad finer pause detection
		'gold-ldcspk': os.path.join(output_dir,'diarised/', f'{fid}-ldc.txt'),
		
		}

	# make output dirs
	rdirs = [list(f.values()) for f in files_dict['control'].values() ] + [list(f.values()) for f in files_dict['patient'].values() ] 
	rdirs = [f for ff in rdirs for f in ff]
	rdirs = set([os.path.dirname(rpath) for rpath in rdirs])

	for rdir in rdirs:
		if rdir and not(os.path.exists(rdir)):
			print('making', rdir)
			os.makedirs(rdir)

	return files_dict




# - - - - - text - - - - -


# parse+normalise a gold (human) transcript
# into standard diarised format
# ret. list of ['Speaker', 'Content in utterance.'] pairs
def parse_transcript(gold_file):

	# line starts should mark dialogue turns
	speaker_labels =  ['s:', 'v:','s\t','v\t', 's;', 'v;',
						   'v ', ' v', ' s', 's ', 'v.', 's.']

	def _spkr(t): # id speaker
		if 's' in t.lower():
			return('S')
		if 'v' in t.lower():
			return('V')
		else:
			return('X')

	def _splitt(ln): # split (speaker, content)
		x = ln.split('\t',1)
		if len(x) == 2:
			return _spkr(x[0]), x[1].replace('\t', ' ')
		else:
			x = ln.split(' ',1)
			return _spkr(x[0]), x[1]

		
	with open(gold_file,'r') as handle:
		xcp = handle.read().splitlines()
	xcp = [l.strip() for l in xcp if l.strip().split()]
	
	parsed = []
	for ln in xcp:
		if ln[:2].lower() in speaker_labels:
			sid, txc = _splitt(ln)
		else:
			sid, txc = 'X', ln
		txc = txc.replace('\t',' ')

		# fix tokenisation for compatibility with forced alignment etc
		# without discarding annotations
		if 'x' not in sid.lower():
			txc = ''.join(c.lower() for c in txc if not c.isdigit())
			txc = re.sub(r'([,\.]+)(\S*\w)', r'\1 \2', txc)
			txc = re.sub(r'([\]\}\)]+)(\S*\w)', r'\1 \2', txc)
			txc = re.sub(r'(\w\S*)([\{]+)', r'\1 \2', txc)
			txc = re.sub(r'\s+([\?\.!,\}\)#=\-–\+~\*"][^\w]*(\s|$))',r'\1',txc)
			txc = re.sub(r'([\{\(])\s+',r'\1',txc)
			txc = re.sub(r'^([#=\-–\+~"\*])\s',r'\1',txc)
			txc = txc.replace('+´', '+ ') #one time
		while '  ' in txc:
			txc = txc.replace('  ', ' ')
		if txc:
			parsed.append([sid,txc.strip()])
			
	return parsed




# read segments from a tsv whose first 3 columns are
# speaker_id, start_time, end_time
# return pyannote annotation
def read_segments_anno(seg_path, col=0):
	with open(seg_path, 'r') as handle:
		segments = handle.read().splitlines()
	segments = [l.split('\t') for l in segments]

	annot = Annotation()
	for l in segments:
		annot[Segment(float(l[1]),float(l[2]))] = l[col]
	return annot
	
	

# - - - - - diarisation - - - - -


# combine neighbouring ordered segments from a single speaker
# if time between is less than minimum pause
def cleanup_segments(segments, min_pause = 0.15):
	#segments = sorted(segments, key = lambda x: (x[0], x[1]))
	
	clean = [segments[0]]
	st = clean[-1][0]
	et = clean[-1][1]
	for s,e,txt in segments[1:]:
	
		assert (s>=st) and (e>=et), f'Segment order: {clean[-1][2]} ---> {txt}'
			
		if s-et >= min_pause:
			clean.append([s,e,txt])
		else:
			st = clean[-1][0]
			txt2 = f'{clean[-1][2]} {txt}'
			txt2 = txt2.replace('  ', ' ')
			clean[-1] = [st,e,txt2]
		et = e
			
		#elif txt:
		#	st = min(clean[-1][0], s)
		#	et = max(et,e)
		#	txt2 = f'{clean[-1][2]} {txt}'
		#	txt2 = txt2.replace('  ', ' ')
		#	print(f'Check order?\n {clean[-1][2]} ---> {txt}')
		#	clean[-1] = [st,et,txt2]

	return clean
	
	
