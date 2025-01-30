import os, subprocess, glob
from pyannote.core import Segment, Timeline, Annotation, notebook
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from acode_util import *

	

	
# ----------------------------
# 
# --- pitch ---
#
# ----------------------------


# returns f0 data as list of Time, F0 if exists, voicing indicator
# bounds set following
# https://www.ling.upenn.edu/courses/Spring_2018/ling620/Nevler2017.pdf
# retry with wider pitch range if percent of voiced frames in file less than min_voicing
# plus 2pass, ref. Hirst The analysis by synthesis of speech melody: from data to models
def get_reaper(wav_path, save_path, reaper_path, max_f0='300', min_f0='75', min_voicing=0.3):


	def _reaper(reaper_path, wav_path, maxf0, minf0):
		f0_raw_data = subprocess.run([reaper_path, "-i", wav_path, '-f', '/dev/stdout', '-x', maxf0, '-m', minf0, '-a'],capture_output=True).stdout
		f0_raw_data = f0_raw_data.decode()
		return f0_raw_data
		
	def _f0s(reaper_output):
		f0_data = reaper_output.split('EST_Header_End\n')[1].splitlines()
		f0_data = [l.split(' ') for l in f0_data] 
		f0_data = [l for l in f0_data if len(l) == 3] # the last line or 2 lines are other info, different format
		f0_data = [ [float(t), float(f), float(v)] for t,v,f in f0_data]
		return f0_data
	
	if os.path.exists(save_path):
		with open(save_path,'r') as handle:
			f0_file_data = handle.read()
			f0_data = _f0s(f0_file_data)
			return f0_data
			

	f0_raw_data = _reaper(reaper_path, wav_path, max_f0, min_f0)
	f0_1pass = _f0s(f0_raw_data)

	
	#print(f' * voiced frames: {sum([v for t,f,v in f0_1pass])/len(f0_1pass):.2f}')
	if sum([v for t,f,v in f0_1pass]) < (len(f0_1pass) * min_voicing):
		print(' F0 tracking failed, retrying with Reaper expanded pitch range')
		f0_raw_data = _reaper(reaper_path, wav_path, '500', '40')
		f0_1pass =  _f0s(f0_raw_data)
		#print(f' * * voiced frames: {sum([v for t,f,v in f0_1pass])/len(f0_1pass):.2f}')
	
	print(f' * voiced frames: {sum([v for t,f,v in f0_1pass])/len(f0_1pass):.2f}')
	#f0_1pass = [f for t,f,v in f0_1pass if v==1]
	f0_1pass = [tfv for tfv in f0_1pass if tfv[2]==1]
	#q1 = np.quantile(f0_1pass,0.25)
	#q3 = np.quantile(f0_1pass,0.75)
	#try:
	#	pfloor = 0.75 * q1
	#	pceil = 1.5 * q3
	#	f0_2pass_data = _reaper(reaper_path, wav_path, str(round(pceil)), str(round(pfloor)))
	#	f0_2pass = _f0s(f0_2pass_data)
	#	assert sum([v for t,f,v in f0_2pass]) > (len(f0_2pass) * min_voicing)
	#except:
	#	pfloor = 0.5 * q1
	#	pceil = 1.5 * q3
	#	print(min(f0_1pass),np.quantile(f0_1pass,0.10),round(pfloor),round(pceil))
	#	f0_2pass_data = _reaper(reaper_path, wav_path, str(round(pceil)), str(round(pfloor)))
	#	f0_2pass = _f0s(f0_2pass_data)
	#	assert sum([v for t,f,v in f0_2pass]) > (len(f0_2pass) * min_voicing)
		
	
	if not os.path.exists(save_path):
		with open(save_path, 'w') as handle:
			handle.write(f0_raw_data)
	
	return f0_1pass


def h2st(hertz,factor):
	return 12 * np.log2(hertz/factor)



# from the whole audio file's pitch track,
# extract pitch for the specified intervals
# then convert it to semitones 
def pitch_data(pitches,speeches):
	
	
	# segment pitch tracks in hz
	hz = [[f for t,f,v in pitches if v==1 and s.overlaps(t)] for s in speeches]
	hz = [s for s in hz if s] # remove segments with no voiced speech
	
	# 90th,10th percentile hz for this speaker globally
	hz90_all = np.percentile([x for seg in hz for x in seg],90)
	hz10_all = np.percentile([x for seg in hz for x in seg],10)
	
	# 90th percentile of each segment in hz
	hz90s = [np.percentile(hzs,90) for hzs in hz]
	
	# 10th percentile of each semgnet in hz
	hz10s = [np.percentile(hzs,10) for hzs in hz]
	
	
	st_global = h2st(hz90_all,hz10_all)
	
	# using each segment's specific 10th percentile as reference point
	st_locals = [h2st(hz90,hz10) for hz90,hz10 in zip(hz90s,hz10s)]
	
	return st_global, st_locals
	


# ----------------------------
# 
# --- rate ---
#
# ----------------------------

def detect_sylls(wav_path, save_path, script_path):
	if not os.path.exists(save_path):
		p = subprocess.run(['octave', script_path, wav_path, save_path]).stdout
	return save_path
	# not working very well...
	
# do this later so file has time to exists
def get_sylls(sylls_path):
	with open(sylls_path, 'r') as handle:
		syl = handle.read().splitlines()
	syl = [l.split(' ') for l in syl]
	syl = [(float(t), float(a)) for t,a in syl]
	return syl


# get the words for each segment
# - pyannote Annotation data structure does not permit 
#   speaker labels and transcribed words in the same object...?
def compile_transcripts(spk_ann,wrd_ann):
	main_speaker = spk_ann.chart()[0][0]
	spk_ann = [(seg,trk,spk) for seg,trk,spk in spk_ann.itertracks(yield_label=True)]
	wrd_ann = [(seg,trk,wrd) for seg,trk,wrd in wrd_ann.itertracks(yield_label=True)]
	wrd_ann = [w for s,w in zip(spk_ann,wrd_ann) if s[2] == main_speaker]
	wrd_ann = [wrd for seg,trk,wrd in wrd_ann]
	return wrd_ann

def syll_per_seg(sylls, segs):
	seg_sylls = [[syll for syll in sylls if s.overlaps(syll[0])] for s in segs]
	return(seg_sylls)


# ----------------------------
# 
# --- speech/pause ---
#
# ----------------------------


# gather measurements from a labelled pyannote Annotation
# TODO:
#  also track pauses following questions
#	e.g. pauses following an Interviewer seg of at least 1.second. \TODO
def compile_segments(pya):
	main_speaker = 'V'#pya.chart()[0][0]
	speeches = []
	pauses = []   
	seg_ends = [(s.end, l) for s,t,l in pya.itertracks(yield_label=True)]
	seg_ends =  sorted(seg_ends, key=lambda x: x[0])
	for segment,track,label in pya.itertracks(yield_label=True):
		if label==main_speaker:
			if speeches: # check preceding pause if this isn't the first segment
				past_turns = [(e,l) for e,l in seg_ends if e<= segment.start]
				last_ended_turn = past_turns[-1]
				if last_ended_turn[1] == main_speaker:
					pause_dur = segment.start - last_ended_turn[0]
					pauses.append(pause_dur)
			speeches.append(segment)

	return speeches, pauses
	

	
# for one recording, return ordered list of:
# 'avg_speech_duration', 'avg_pause_duration', 'pauses_per_minute', 
#  'percent_speaking', 'speech_pause_ratio', 'pitch_range_global', 
#  'words_per_speech', 'word_duration', 'syllables_per_speech', 'syllable_duration'
def compile_features(feats):
	#feats['syllables']
	#conversation_time_span = speeches[-1].end - speeches[0].start
	# speech_dur = pya.chart()[0][1]
	speech_dur = sum([s.duration for s in feats['speeches']])
	avg_speech_dur = np.mean([s.duration for s in feats['speeches']])
	avg_pause_dur = np.mean(feats['pauses'])
	speaker_time_span = speech_dur + sum(feats['pauses'])
	n_pause_per_minute = len(feats['pauses'])/(speaker_time_span/60)
	percent_speech = speech_dur/speaker_time_span*100
	speech_pause_ratio = speech_dur/sum(feats['pauses'])
	#avg_pitch_range_perseg = np.mean(ranges[1])
	
	# don't count asr errors with no words/sylls exist in segment
	word_per_seg = np.mean([len(w.split(' ')) for w in feats['words'] if w])
	syll_per_seg = np.mean([len(s) for s in feats['syllables'] if s])
	
	seg_sylls = zip(feats['syllables'],[s.duration for s in feats['speeches']])
	seg_words = zip(feats['words'],[s.duration for s in feats['speeches']])
	seg_sylls = [(s,d) for s,d in seg_sylls if s]
	seg_words = [(w,d) for w,d in seg_words if w]
	syll_dur = sum([d for s,d in seg_sylls])/sum([len(s) for s,d in seg_sylls])
	word_dur = sum([d for s,d in seg_words])/sum([len(w.split(' ')) for w,d in seg_words])
	
	#total_sylls = sum([len(s) for s in feats['syllables']])
	#total_words = sum([len(w) for w in feats['words']])
	#word_dur = speech_dur/total_words
	#syll_dur = speech_dur/total_sylls
	
	return avg_speech_dur, avg_pause_dur, n_pause_per_minute,\
	 percent_speech, speech_pause_ratio, feats['global_f0_range'], word_per_seg, word_dur, syll_per_seg, syll_dur
	 



def featurise_one(files_data,pitch_extracter,syll_detector):

	wav_file = files_data['wav']
	segment_file = files_data['segmented']
	f0_file = files_data['f0']
	syll_file = files_data['syl']
	
	print(f'Finding features for sample {fn(wav_file)}')
	tmp_wav = wav16mono(wav_file,files_data['tmp-wav'])
	pitch_track = get_reaper(tmp_wav, f0_file, pitch_extracter)
	syllable_points = detect_sylls(tmp_wav, syll_file, syll_detector)
	
	segments = read_segments_anno(segment_file)
	transcripts = compile_transcripts(segments,read_segments_anno(segment_file,3))
	speeches, pauses = compile_segments(segments)
	range_global, ranges_local = pitch_data(pitch_track, speeches)
	#syllables = syll_per_seg(syllable_points, speeches)
	
	assert len(transcripts) == len(speeches)
	
	return {'segments': segments, 'speeches': speeches, 'pauses': pauses, 'global_f0_range': range_global, 'local_f0_ranges': ranges_local, 'syllables': syllable_points, 'words': transcripts, 'PGroup': files_data['group']}
	



def featurise_speech(corpus_dir,features_dir,segmentation):

	data_files = setup_nextcloud_outputs(corpus_dir, features_dir)
	
	# executable from installing reaper
	# https://github.com/google/REAPER/
	# TODO Praat instead of reaper
	f0_extractor = "./localmodels/REAPER/build/reaper"
	
	# from http://languagelog.ldc.upenn.edu/myl/findsylls
	# (slightly edited to rename output file)
	syll_detector = './localmodels/findsylls.octave'
		
	
	feats_data = {'patient': {}, 'control': {}}
	for group in ['control','patient']:
		for record in data_files[group].keys():
		
			data_files[group][record]["segmented"] = data_files[group][record][segments_using]
			
			feats_data[group][record] = featurise_one(data_files[group][record],f0_extractor,syll_detector)
	
	
	# second round so files have time to exists...
	for group in ['control','patient']:
		for record in feats_data[group].keys():
			try:
				feats_data[group][record]['syllables'] = syll_per_seg(get_sylls(feats_data[group][record]['syllables']),feats_data[group][record]['speeches'] )
			except:
				print(f"the octave file for: {record} probably still doesn't have contents, it happens sometimes. delete the empty 0kb .sylls file and try re-running the whole thing, it will usually work fine then.")
				raise
				
	return feats_data
	
	
	


def report_features(features_data, output_file):
	
	if not os.path.exists(os.path.dirname(output_file)):
		os.makedirs(os.path.dirname(output_file))
	
	with open(output_file,'w') as handle:
		handle.write('\t'.join(['sample_id', 'group', 'avg_speech_duration', 'avg_pause_duration', 'pauses_per_minute', 'percent_speaking', 'speech_pause_ratio', 'pitch_range_global', 'words_per_speech', 'word_duration'\
		, 'syllables_per_speech', 'syllable_duration'\
		])+'\n')


	for group in ['control','patient']:
		for record in features_data[group].keys():
			feats_info = compile_features(features_data[group][record])
			
			with open(output_file,'a') as handle:
				handle.write(f'{record}\t{features_data[group][record]["PGroup"]}\t'+'\t'.join([str(round(f,4)) for f in feats_info])+'\n')

	return output_file




if __name__ == "__main__":

	audio_corpus_dir = '/home/cati/proj/acode/NextCloud/Data/'
	feats_save_dir = '../../acoustic-processing/output-is020/'
	segments_using = 'gold-ldcspk' # gold-ldcspk or gold-cfaspk
	
	output_file = os.path.join(feats_save_dir,'ACOUSTIC_FEATURES-force-with-ldc.tsv')
	
	feature_data = featurise_speech(audio_corpus_dir,feats_save_dir,segments_using)
	
	feature_info = report_features(feature_data, output_file)
	
 
	  
