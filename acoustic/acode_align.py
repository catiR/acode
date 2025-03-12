import os, re, json, subprocess
import torch
from ctc_forced_aligner import (
	load_audio,
	load_alignment_model,
	generate_emissions,
	preprocess_text,
	get_alignments,
	get_spans,
	postprocess_results,
)
from pyannote.core import Segment, Timeline, Annotation
from acode_util import *



# prepare human transcript for ctc-forced-aligner
# - discard most punctuation marks but keep sentence tokenisation
# - fill in special <star> alignment token
# set keep_guesses False to replace more unclear transcripts with <star>
def ctc_prep_transcript(gold_file, norm_file,star_char='🥭',keep_guesses = True):
	orig = parse_transcript(gold_file)

	def inside_replacer(span,excludes):
		span = ''.join([c for c in span.group(0) if c not in excludes])
		span = span.split()
		span = ' '.join([star_char for word in span])
		return span

	def bracket_repl(span):
		return inside_replacer(span, '{}')
	
	def paren_repl(span):
		return inside_replacer(span, '()')

	def _line_prep(l,keep_guesses = True):
		ln = re.sub(r'\{[^}]*\}+', bracket_repl, l)
		if not keep_guesses:
			ln = re.sub(r'\([^)]*\)+', bracket_repl, ln)
		ln = [re.sub(r"[-–=#~+\{\}'\[\]\(\)\*]", '', w) or star_char
				  for w in ln.split(' ')]
		ln = ' '.join(ln)
		assert (len(ln.split(' ')) == len(l.split(' '))) and ('  ' not in ln)
		return ln

	prep_lines = [star_char] + [_line_prep(t,keep_guesses).strip()
			 for s,t in orig if s.lower() != 'x'] + [star_char]
	
	#with open(norm_file,'w') as handle:
	#	handle.write('\n'.join(prep_lines))
	return prep_lines



# add ctc-forced-alignment timings
# to gold labelled transcripts
# output is readable by ELAN
def label_speakers_alignments(gold_file,aligns_file,cfa_save_file,
								  lab_file=None,cfaldc_save_file=None,
								  pya_file=None, cfapya_save_file=None):
								  
	aligner_skip_char='🥭'
	orig = parse_transcript(gold_file)
	

	with open(aligns_file, 'r') as handle:
		aligns = json.load(handle)
	aligns = aligns['segments']
	aligns = sorted(aligns, key=lambda x: x['start'])
		

	gold_seq = []
	for spk, turn in orig:
		if 'x' in spk.lower():
			gold_seq.append([spk, turn])
		else:
			for word in turn.split(' '):
				if word:
					gold_seq.append([spk, word])

	labelled_words = [['X',0.0, 1.0,'']] #dummy segment with times, remove later
	for spk, txt in gold_seq:
		if 'x' in spk.lower():
			labelled_words.append([spk, labelled_words[-1][1],
									   labelled_words[-1][2], txt])
		else:
			aln = aligns[0]
			if aligner_skip_char not in aln['text']:
				achar = ''.join(x.lower() for x in aln['text'] if x.isalpha())
				gchar = ''.join(x.lower() for x in txt if x.isalpha())
				assert achar == gchar
			labelled_words.append([spk, aln['start'], aln['end'], txt])
			aligns = aligns[1:]

	labelled_words = [x for x in labelled_words if x[3]]
	assert not aligns


	final_cfa_segments, final_ldc_segments, final_pya_segments = [], [], []
	for speaker in set([spk for spk,s,e,w in labelled_words]):
		spwds = [[s,e,w] for spk,s,e,w in labelled_words if speaker==spk]
		
		if lab_file:
			if 'x' in speaker.lower():
				spwds_ldc = spwds
			else:
				spwds_ldc = relabel_sd(spwds,lab_file)
			spsegs_ldc = cleanup_segments(spwds_ldc)
			for s,e,w in spsegs_ldc:
				final_ldc_segments.append([speaker,s,e,w])
				
		if pya_file:
			if 'x' in speaker.lower():
				spwds_pya = spwds
			else:
				spwds_pya = relabel_sd(spwds,pya_file)
			spsegs_pya = cleanup_segments(spwds_pya)
			for s,e,w in spsegs_pya:
				final_pya_segments.append([speaker,s,e,w])
		
		spsegs = cleanup_segments(spwds)
		for s,e,w in spsegs:
			final_cfa_segments.append([speaker,s,e,w])
	
	for finalsegments, savefile in ([(final_cfa_segments,cfa_save_file),
		(final_ldc_segments,cfaldc_save_file),
		(final_pya_segments,cfapya_save_file)]):
		if finalsegments:
			finalsegments = sorted(finalsegments, key=lambda x: x[1])
			finalsegments = [f'{sk}\t{st}\t{et}\t{txt}' for sk,st,et,txt in finalsegments]
	
		if not os.path.exists(savefile):
			with open(savefile,'w') as handle:
				handle.write('\n'.join(finalsegments))
	
	return cfa_save_file 




# try to intersect forced alignments
# with ldc-sad or pyannote-vad speech activity
# as far as possible
def relabel_sd(word_times, sd_path):

	# overlap
	# because & comparison for segments
	# hadles zero duration segments wrong for this case
	# at least 1 input is nonzero dur though
	def _ovls(s1, s2):
		if (not s1) or (not s2):
			zro, itv = sorted([s1,s2], key = lambda x: x.duration)
			# define that they overlap if point is inside or equal to starttime
			if (zro.start < itv.start) or (zro.end >= itv.end):
				return Segment(0,0).duration # this evals to False/not exist
			else:
				return 0.1 # why doesnt pyannote have anything less awful?
		else:
			return (s1 & s2).duration
			
			
	# get the best fit interval for a forcealigned word
	# among all available intervals from a speech detector
	# return pyannote segment
	def get_best_interval(word,all_intervals):
		candidates = [(inv, _ovls(inv,word)) for inv in all_intervals if _ovls(inv,word)]
		if not candidates:
			# if word doesn't overlap any VAD segment at all,
			# keep its forced aligner timing
			return word
		elif len(candidates) == 1:
			# word overlaps exactly one segment, return it
			return candidates[0][0]
		else:
			# word overlaps multiple segments
			# return segment containing largest percentage of word
			# if multiple tied (0 duration word, 
			#			or overlapping segments both contain whole word)
			#   return shortest of the tied segments
			# (this case probably shouldn't happen anymore, but)
			best = max(candidates, key = lambda x: (x[1],1/(x[0].duration)))
			return best[0]
			
			
	# get speech activity detection/segmentation
    
	if sd_path[-3:] == 'lab':
		sd = 'ldc'
	else:
		sd = 'pya'

	seg_tl = Timeline()
	
	with open(sd_path, 'r') as handle:
		sds = handle.read().splitlines()
	sds = [l.split('\t') for l in sds]
	
	if sd == 'ldc':
		sds = [l for l in sds if 'non-speech' not in l]
		for s,e,_ in sds:
			seg_tl.add(Segment(float(s),float(e)))
	else: # must be pya
		for _,s,e in sds:
			seg_tl.add(Segment(float(s),float(e)))
		# enforce no overlapping segments
		# since this is one speaker at a time anyway
		seg_tl = seg_tl.segmentation()
	
	
	# get forced alignments
	word_times = [(Segment(float(s),float(e)), w) for s, e, w in word_times]
	
	# for every word in the transcript, find its best interval
	word_intervals = [(get_best_interval(tim,seg_tl),wd) for tim,wd in word_times]
	
	# ?word order ok
	for i in range(1, len(word_intervals)):
		assert word_intervals[i][0] >= word_intervals[i-1][0]
	
	# for every interval, 
	#  get all of its words in their correct order
	seg_tl = Timeline(segments = [seg for seg,wd in word_intervals] )
	retimed_text = []
	for seg in seg_tl:
		txts = ' '.join([wd for itv,wd in word_intervals if itv==seg])
		retimed_text.append([seg.start, seg.end, txts])
		
	return retimed_text
	



# do forced alignment for long audios
# https://github.com/MahmoudAshraf97/ctc-forced-aligner
# (but install the one from https://github.com/catiR/ctc-forced-aligner)
def align_acode(data_files, save_dir, aln_model = 'is', save_spec = ''):

	assert aln_model.lower() in ['is', 'mul'], ('Model spec must be is [icelandic]'
									 'or mul [tilingual]')

	###### load alignment models ######

	if aln_model == 'is':
		model_path = ("language-and-voice-lab/"
						  "wav2vec2-large-xlsr-53-icelandic-ep30-967h")
		romanize = False
		attention_implementation = 'pad'
		# this model doesnt return attention mask, so pad input only
		context_length = 0.20

	elif aln_model == 'mul':
		# this has attention and Star emit, REQUIRES romanize=True
		model_path = "MahmoudAshraf/mms-300m-1130-forced-aligner"
		romanize = True
		context_length = 2.0
		attention_implementation = None #not actually none, uses a default

	
	language = "isl" # ISO-639-3 Language code
	device = "cuda" if torch.cuda.is_available() else "cpu"
	batch_size = 2 #16
	window_length=30
	merge_threshold = 0.0 # minimum silence duration 150 ms in theory,
	                      #but even at 0 it rarely finds pauses
	star_char = '🥭' # reserved character, shouldn't have been used in transcripts
	star_frequency = "custom" # required for acode!!

	alignment_model, alignment_tokenizer = load_alignment_model(
		device,
		model_path = model_path,
		dtype=torch.float16 if device == "cuda" else torch.float32,
		attn_implementation = attention_implementation,
		)
	

	
	for fid, finfo in data_files.items():
	
		# things that need to already exist:
		# - source(gold) transcript from nextcloud
		# - wav from nextcloud
		# - ldc bpcsad speech activity segmentation
		# - pyannote diarisation
		source_xcp_path = finfo[1]
		orig_wav = finfo[0]
		ldc_lab_path = os.path.join(save_dir,'feats','lab',f'{fid}.lab')
		pya_dia_path = os.path.join(save_dir,'diarised','pya',f'{fid}.txt')
		
		# more things the aligner will need
		# - normalised trancript
		# - 16khz mono wav (shouldnt actually be necessary 
		#     but it probably already exists anyway)
		source_normed_path = os.path.join(save_dir,f'align{save_spec}',f'{fid}.cfanorm')
		tmp_wav = os.path.join(save_dir,'tmp',f'{fid}.wav')
		
		# outputs
		# - ctc-forced-aligner word alignments (no speaker labels)
		# - timed diarised transcript based on cfa alignments,
		#     diarisation uses each word's speaker labels in source transcript
		# - timed diarised transcript using ldc rather than cfa where possible 
		#       to know *when* someone was speaking,
		#     but uses cfa to identify *who* the speaker was
		# - timed diarised transcript using pyannote to know when is speech,
		#     but still labels speaker identity through cfa alignments
		#     ( = ignores pyannote's diarisation)
		alignment_json_path = os.path.join(save_dir,f'align{save_spec}',f'{fid}.json')
		cfa_dia_path = os.path.join(save_dir,'diarised',
								f'cfa{save_spec}',f'{fid}.txt')
		cfaldc_dia_path = os.path.join(save_dir,'diarised',
								f'cfa_ldc{save_spec}',f'{fid}.txt')
		cfapya_dia_path = os.path.join(save_dir,'diarised',
								f'cfa_pya{save_spec}',f'{fid}.txt')
		
		for writepath in [source_normed_path, tmp_wav, alignment_json_path,
					cfa_dia_path, cfaldc_dia_path,cfapya_dia_path]:
			mds(writepath)
		
		
		###### run forced alignment ######
		if os.path.exists(alignment_json_path):
			print(f"Didn't overwrite existing forced alignment for {fid}.")
			print('To redo alignment with different parameters, '
					'provide a different output directory.')
					
		else:
			print(f'MA-ctc-force-aln forced alignment, {fid}')
			xcp = ctc_prep_transcript(source_xcp_path, source_normed_path)
			xcp = ' '.join(xcp)

			tokens_starred, text_starred = preprocess_text(xcp,
				romanize=romanize, language=language,
				star_frequency=star_frequency, star_char = star_char)

			audio_waveform = load_audio(tmp_wav,
							alignment_model.dtype, alignment_model.device)
			
			emissions, stride = generate_emissions(alignment_model,
				audio_waveform,	window_length=window_length,
				context_length=context_length, batch_size=batch_size)

			segments, scores, blank_token = get_alignments(emissions,
				tokens_starred, alignment_tokenizer, star_char = star_char)

			spans = get_spans(tokens_starred, segments, blank_token,
								  star_char = star_char)
			
			timestamps = postprocess_results(text_starred, spans, stride,
								scores, merge_threshold = merge_threshold)

			
			with open(alignment_json_path, "w") as handle:
				json.dump({"text": xcp[1:-1].strip(), "segments": timestamps[1:-1]},
							  handle, indent=4)
			
			
		###### timed diarised transcripts for ELAN ######

		_ = label_speakers_alignments(source_xcp_path, alignment_json_path,
			cfa_dia_path, 
			ldc_lab_path, cfaldc_dia_path,
			pya_dia_path, cfapya_dia_path)


if __name__ == "__main__":
	
	original_data_dir = '../../NextCloud/Data/'
	save_dir = f'../../acoustic-processing/output-is020/'
	
	data_files = compile_nextcloud_files(original_data_dir)

	align_acode(data_files, save_dir, aln_model = 'is')
	
