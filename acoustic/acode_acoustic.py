import os, subprocess, glob
from pyannote.core import Segment, Timeline, Annotation
import numpy as np
#import matplotlib.pyplot as plt
#import scipy.stats as stats
from acode_util import *

	

# convert hertz value to semitones
# factor is speaker's 10th percentile in hz
def h2st(hertz,factor):
	return 12 * np.log2(hertz/factor)


def read_voicesauce_pitch(pitch_file):
	with open(pitch_file, 'r') as handle:
		f0 = handle.read().splitlines()
	f0 = [l.split('\t')[1:] for l in f0[1:]]
	f0 = [(float(t)/1000, float(f)) for t,f in f0 if f.lower() != 'nan']
	return f0



# gather one speaker's speech + pauses 
# from a labelled pyannote Annotation
# Nevler 2017:
# " Silent pauses were excluded from analysis if they were at the beginning
#   or end of the audio or immediately following interviewer prompting."
def compile_speaker_segments(pya, main_speaker = 'V'):

	if not main_speaker:
		# usually but not always finds the right person
		# so try not to use main_speaker=None
		main_speaker = pya.chart()[0][0]
		
	speeches = []
	pauses = []   
	seg_ends = [(s.end, l) for s,t,l in pya.itertracks(yield_label=True)]
	seg_ends =  sorted(seg_ends, key=lambda x: x[0])
	
	for speaking,track,label in pya.itertracks(yield_label=True):
		if label==main_speaker:
			if speeches: # check preceding pause if this isn't the first segment
				past_turns = [(e,l) for e,l in seg_ends if e<= speaking.start]
				last_ended_turn = past_turns[-1]
				if last_ended_turn[1] == main_speaker:
					pause_dur = speaking.start - last_ended_turn[0]
					pauses.append(pause_dur)
			speeches.append(speaking)

	return speeches, pauses
	



# nevler et al 2o17
# "We extracted f0 estimates for the 10th through the 90th f0 percentiles 
#     for each speech segment 
#   and then calculated the mean f0 for each 10 percentile bin per participant."
def pitch_data(pitches,speeches):
	
	# segment pitch tracks in hz
	hz_segments = [[f for t,f in pitches if s.overlaps(t)] for s in speeches]
	hz_segments = [s for s in hz_segments if s] # remove segments with no voiced speech
	
	
	# participant's overall 10th percentile
	# -- do not use for normalisation when replicating,
	# reported average 10th percentile across segments is zero
	# so need to normalise hz by average 10th percentile across segments
	# not global 10th percentile.
	hz10_global = np.percentile([f for seg in hz_segments for f in seg],10)
	
	# "f0 estimates for the 10th through the 90th f0 percentiles 
#     for each speech segment"
	hz_percentiles = [{p:np.percentile(hzs, p) for p in range(10,100,10)} 
						for hzs in hz_segments ]
	
	# "the mean f0 for each 10 percentile bin "
	hz_binmeans = {p: np.mean([s[p] for s in hz_percentiles]) 
					for p in range(10,100,10)}
	
	# use this factor to normalise so segment-averaged results are zeroed			
	hz10_NORM = hz_binmeans[10]
	st_binmeans = {p:h2st(hz,hz10_NORM) for p,hz in hz_binmeans.items()}

	assert st_binmeans[10] == 0
	
	return hz10_NORM, st_binmeans
	



def featurise_session(segment_file, pitch_file):

	segments = read_diarisation(segment_file)

	speeches, pauses = compile_speaker_segments(segments)
	
	# meanSpch	mean speech segment duration in seconds
	meanSpch = np.nanmean([s.duration for s in speeches])
	
	# meanPause mean pause segment duration in seconds
	meanPause = np.nanmean(pauses)
	
	# totalSpch	total SUM of all speech segments in seconds (not normalized to recording length)			
	totalSpch = sum([s.duration for s in speeches])
	
	# pause_rate (pauseNum/totalSpch)*60 Number of pause events per minute of speech
	pauseNum = len(pauses)
	pause_rate = (pauseNum/totalSpch)*60
								
	# total_time Total speech + pause time (excluding interviewer speech)
	# (not normalized to recording length)  in seconds
	total_time = totalSpch + sum(pauses)
	
	# percent_spch (totalSpch/totaltime)*100 
	# Percentage of time spent speaking vs. pausing; 100-percent_spch = %Pause	
	percent_spch = 100*totalSpch/total_time
	
	
	pitch_track = read_voicesauce_pitch(pitch_file)
	
	# f0med	MEDIAN f0 Range (50th percentile) in semitones
	#     --> f0_st[50]				
	# f0range TRIMMED pitch range, 90th percentile pitch in semitones
	#     --> f0_st[90]
	f0_10_hz, f0_st = pitch_data(pitch_track, speeches)
	
	
	# InterviewerNumSegments COUNT of interviewer speech segments, number of interviewer prompts
	iv_speeches, _ = compile_speaker_segments(segments, main_speaker = 'S')
	interviewerNum = len(iv_speeches)
	# interviewerNumTurns also in output
	# based on counting turns in text transcription
	
	
	acodes = {'meanSpch': meanSpch, 
				'meanPause': meanPause,
				'totalSpch': totalSpch,
				'pause_rate': pause_rate,
				'total_time': total_time,
				'percent_spch': percent_spch,
				'f0_10_hz': f0_10_hz,
				'interviewerNumSegments':interviewerNum}
				
	for p,st in f0_st.items():
		acodes[f'f0_{p}_st'] = st
	
	return acodes



def compile_acode_featurisation(segment_dir,f0_dir,original_corpus_dir,save_file):

	mds(save_file)
	
	output_columns = ['record_id', 'group', 'cohort',
						'meanSpch', 'meanPause', 'pause_rate',
						'totalSpch','total_time','percent_spch',
						'interviewerNumSegments', 'interviewerNumTurns',
						'f0_10_hz']
						
	for p in range(10,100,10):
		output_columns.append(f'f0_{p}_st')

	with open(save_file,'w') as handle:
		handle.write('\t'.join(output_columns))
		

	original_files = compile_nextcloud_files(original_corpus_dir)
	segment_files = glob.glob(segment_dir+'*.txt')
	for segment_path in sorted(segment_files):
	
		pitch_path = os.path.join(f0_dir,f'{fn(segment_path)}.tsv')
		
		acode_features = featurise_session(segment_path, pitch_path)
		
		acode_features['record_id'] = fn(segment_path)
		acode_features['group'] = original_files[fn(segment_path)][2]
		acode_features['cohort'] = original_files[fn(segment_path)][3]
		
		# alternate version of InterviewerNum 
		# COUNT of interviewer speech segments, number of interviewer prompts
		# this version based on transcribed turns
		transcript = original_files[fn(segment_path)][1]
		transcript = parse_transcript(transcript)
		acode_features['interviewerNumTurns'] = len([l for l in transcript if l[0]=='S'])
		
		output_row = [acode_features[variable] for variable in output_columns[:3]]
		output_row += [str(round(acode_features[variable],4)) for variable in output_columns[3:]]
		with open(save_file,'a') as handle:
			handle.write('\n'+'\t'.join(output_row))
			
	return save_file




if __name__ == "__main__":

	acoustic_data_dir = '../../acoustic-processing/output-is020/'
	
	# only used for alternate interviewer segments count--
	original_corpus = '/home/cati/proj/acode/NextCloud/Data/'
	
	# praat pitch algorithm
	# ac - autocorrelation
	# cc - crosscorrelation
	ppa = 'ac'
	
	# selected diarisation method
	# cfa_ldc: gold transcripts aligned by ctc-forced-aligner 
	#            to identify when each person is speaking
	#          and merged with ldc speech activity detection
	# cfa : just ctc-forced-aligner times - don't use this
	#       it undersegments so badly that some speakers have 0 pauses
	# pya : pyannote automatic segmentation and diarisation
	# pya_ldc:  ldc speech segmentation with pyannote speaker labels
	dia = 'ldc'
	
	# diarisation dir, see acode_align.py
	# 4 column tsv ELAN transcript
	# Speaker_id, Start_time, End_time, Text
	# speaker_ids in {S, V, X},
	# V_iðmælandi is data analysed, X is ignored nonspeech/annotation
	segmentation_dir = f'{acoustic_data_dir}diarised/{dia}/'
	
	# pitch tracking, see extract_acoustic.py
	pitch_dir = f'{acoustic_data_dir}feats/voicesauce/praat/{ppa}/'
	syllable_detect_dir = None # syllable detection not used this version
	

	feature_output = f'{acoustic_data_dir}/ACODE/ACOUSTIC_FEATURES-praat-{ppa}--{dia}.tsv'
	
	
	_ = compile_acode_featurisation(segmentation_dir,pitch_dir,original_corpus,feature_output)
	




