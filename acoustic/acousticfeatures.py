import os, subprocess, glob
from pyannote.core import Segment, Timeline, Annotation, notebook
import numpy as np
from scripts.util import *

	

	
# ----------------------------
# 
# --- pitch ---
#
# ----------------------------

# TODO:
# praat instead of reaper.


# returns f0 data as list of Time, F0 if exists, voicing indicator
# bounds set following
# https://www.ling.upenn.edu/courses/Spring_2018/ling620/Nevler2017.pdf
# retry with wider pitch range if percent of voiced frames in file less than min_voicing
def get_reaper(wav_path, save_path, reaper_path, maxf0='300', minf0='75', min_voicing=0.3):


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
			f0_raw_data = handle.read()
	else:
		f0_raw_data = _reaper(reaper_path, wav_path, maxf0, minf0)
	
	f0_data = _f0s(f0_raw_data)
	
	print(f' * voiced frames: {sum([v for t,f,v in f0_data])/len(f0_data):.2f}')
	if sum([v for t,f,v in f0_data]) < (len(f0_data) * min_voicing):
		print(' F0 tracking failed, retrying with Reaper default pitch range')
		f0_raw_data = _reaper(reaper_path, wav_path, '500', '40')
		f0_data =  _f0s(f0_raw_data)
		print(f' * * voiced frames: {sum([v for t,f,v in f0_data])/len(f0_data):.2f}')
	
	if not os.path.exists(save_path):
		with open(save_path, 'w') as handle:
			handle.write(f0_raw_data)
	
	return f0_data


def h2st(hertz,factor):
	return 12 * np.log2(hertz/factor)


# from the whole audio file's pitch track,
# extract pitch for the specified intervals
# then convert it to semitones 
def pitch_data(pitches,speeches):
	
	# segment pitch tracks in hz
	hz = [[f for t,f,v in pitches if v==1 and s.overlaps(t)] for s in speeches]
	hz = [s for s in hz if s] # remove segments with no voiced speech
	
	# 90th percentile of each segment in hz
	hz90s = [np.percentile(hzs,90) for hzs in hz]
	
	# 10th percentile of each semgnet in hz
	hz10s = [np.percentile(hzs,10) for hzs in hz]
	
	# 10th percentile hz for this speaker globally
	hz10_all = np.percentile([x for seg in hz for x in seg],10)
	
	
	#TODO: 
	# which is correct, global or self?
	
	# convert each segment's 90th percentile to semitones 
	#  using speaker's overall 10th percentile as reference point
	st_global = [h2st(hz90,hz10_all) for hz90 in hz90s]
	
	# using each segment's specific 10th percentile as reference point
	st_self = [h2st(hz90,hz10) for hz90,hz10 in zip(hz90s,hz10s)]
	
	return st_global, st_self
	



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
	main_speaker = pya.chart()[0][0]
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
	





def compile_features(speeches, pauses, ranges):
	time_span = speeches[-1].end - speeches[0].start
	# speech_dur = pya.chart()[0][1]
	speech_dur = sum([s.duration for s in speeches])
	avg_speech_dur = np.mean([s.duration for s in speeches])
	avg_pause_dur = np.mean(pauses)
	n_pause_per_minute = len(pauses)/(time_span/60)
	percent_speech = speech_dur/time_span*100
	
	#TODO which one:
	avg_pitch_range_global = np.mean(ranges[0])
	avg_pitch_range_perseg = np.mean(ranges[1])


	return avg_speech_dur, avg_pause_dur, n_pause_per_minute,\
	 percent_speech, avg_pitch_range_global, avg_pitch_range_perseg
	 



# return ordered list of:
# 'avg_speech_duration', 'avg_pause_duration', 'pauses_per_minute', 
#  'percent_speaking', 'pitch_range_global', 'pitch_range_local'
def featurise_one(wav_file,segment_file,f0_file,pitch_extracter):

	print(f'Finding features for sample {fn(wav_file)}')
	tmp_wav = wav16mono(wav_file)
	pitch_track = get_reaper(tmp_wav, f0_file, pitch_extracter)
	
	segments = read_segments_anno(segment_file)
	speeches, pauses = compile_segments(segments)
	ranges_a, ranges_b = pitch_data(pitch_track, speeches)
	
	sd, pd, np, cs, r_a,r_b = compile_features(speeches, pauses, (ranges_a,ranges_b))

	return [sd, pd, np, cs, r_a,r_b]
	
	

	
	
def featurise_speech():
	original_data_dir = '/home/cati/proj/acode/NextCloud/Data/'
	save_dir = './acoustic-output/'
	
	# executable from installing reaper
	# https://github.com/google/REAPER/
	# TODO Praat instead of reaper
	f0_extractor = "./localmodels/REAPER/build/reaper"
	
	output_file = (os.path.join(save_dir,'Results/ACOUSTIC_FEATURES.tsv'))
	if not os.path.exists(os.path.dirname(output_file)):
		os.makedirs(os.path.dirname(output_file))
	data_files = find_nextcloud_files(original_data_dir, save_dir)
	
	
	with open(output_file,'w') as handle:
		handle.write('\t'.join(['sample_id', 'group', 'avg_speech_duration', 'avg_pause_duration', 'pauses_per_minute','percent_speaking','pitch_range_global','pitch_range_local'])+'\n')
	
	
	#for group in ['control','patient']:
	for group in ['patient']:
		for speaker,file_paths in data_files[group].items():
			feats = featurise_one(file_paths['wav'],file_paths['pya'],file_paths['f0'],f0_extractor)
			with open(output_file,'a') as handle:
				handle.write(f'{speaker}\t{str.title(group)}\t'+'\t'.join([str(round(f,4)) for f in feats])+'\n')
	
	



if __name__ == "__main__":
	featurise_speech()
	
 
	  
