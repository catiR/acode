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
def ctc_prep_transcript(gold_file, norm_file,keep_guesses = True):
	orig = parse_transcript(gold_file)

	def inside_replacer(span,excludes):
		span = ''.join([c for c in span.group(0) if c not in excludes])
		span = span.split()
		span = ' '.join(['<star>' for word in span])
		return span

	def bracket_repl(span):
		return inside_replacer(span, '{}')
	
	def paren_repl(span):
		return inside_replacer(span, '()')

	def _line_prep(l,keep_guesses = True):
		ln = re.sub(r'\{[^}]*\}+', bracket_repl, l)
		if not keep_guesses:
			ln = re.sub(r'\([^)]*\)+', bracket_repl, ln)
		ln = [re.sub(r"[-–=#~+\{\}'\[\]\(\)\*]", '', w) or '<star>'
				  for w in ln.split(' ')]
		ln = ' '.join(ln)
		assert (len(ln.split(' ')) == len(l.split(' '))) and ('  ' not in ln)
		return ln

	prep_lines = [_line_prep(t,keep_guesses).strip()
					  for s,t in orig if s.lower() != 'x']
	
	with open(norm_file,'w') as handle:
		handle.write('\n'.join(prep_lines))
	return prep_lines



# apply ldc-bpcsad speech activity detector
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



# add ctc-forced-alignment timings
# to gold labelled transcripts
# output is readable by ELAN
def label_speakers_alignments(gold_file,aligns_file,cfa_save_file,
								  lab_file=None,ldc_save_file=None):
								  
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
			if '<star>' not in aln['text']:
				achar = ''.join(x.lower() for x in aln['text'] if x.isalpha())
				gchar = ''.join(x.lower() for x in txt if x.isalpha())
				assert achar == gchar
			labelled_words.append([spk, aln['start'], aln['end'], txt])
			aligns = aligns[1:]

	labelled_words = [x for x in labelled_words if x[3]]
	assert not aligns


	final_cfa_segments, final_ldc_segments = [], []
	for speaker in set([spk for spk,s,e,w in labelled_words]):
		spwds = [[s,e,w] for spk,s,e,w in labelled_words if speaker==spk]
		
		if lab_file:
			if 'x' in speaker.lower():
				spwds_ldc = spwds
			else:
				spwds_ldc = relabel_ldc(spwds,lab_file)
			spsegs_ldc = cleanup_segments(spwds_ldc)
			for s,e,w in spsegs_ldc:
				final_ldc_segments.append([speaker,s,e,w])
		
		spsegs = cleanup_segments(spwds)
		for s,e,w in spsegs:
			final_cfa_segments.append([speaker,s,e,w])
	
	for finalsegments, savefile in ([(final_cfa_segments,cfa_save_file),
		(final_ldc_segments,ldc_save_file)]):
		if finalsegments:
			finalsegments = sorted(finalsegments, key=lambda x: x[1])
			finalsegments = [f'{sk}\t{st}\t{et}\t{txt}' for sk,st,et,txt in finalsegments]
	
		if not os.path.exists(savefile):
			with open(savefile,'w') as handle:
				handle.write('\n'.join(finalsegments))
	
	return cfa_save_file #, ldc_save_file




# try to intersect forced alignments
# with ldc's speech activity
# as far as possible
def relabel_ldc(word_times, lab_path):

	ldc_tl = Timeline()
	
	with open(lab_path, 'r') as handle:
		ldcs = handle.read().splitlines()
		
	ldcs = [l.split('\t') for l in ldcs]
	ldcs = [l for l in ldcs if 'non-speech' not in l]
	for s,e,_ in ldcs:
		ldc_tl.add(Segment(float(s),float(e)))
		
	word_times = [(Segment(float(s),float(e)), w) for s, e, w in word_times]
	
	# overlap
	# because & comparison for segments
	# hadles zero duration segments wrong for this case
	# at least 1 input is nonzero dur though
	def _ovls(s1, s2):
		if (not s1) or (not s2):
			zro, itv = sorted([s1,s2], key = lambda x: x.duration)
			# define that they overlap if point is inside or equal to starttime
			if (zro.start < itv.start) or (zro.end >= itv.end):
				return Segment(0,0).duration
			else:
				return 0.1 # why doesnt pyannote have anything less awful?
		else:
			return (s1 & s2).duration

	
	# is the query interval the best fit interval for this word
	# among all available intervals
	def is_best_interval(word,query,all_intervals):
		word = word[0]
		candidates = [(inv, _ovls(inv,word)) for inv in all_intervals if _ovls(inv,word)]
		if len(candidates) == 1:
			return True
		else:
			best = max(candidates)#, key = lambda x: x[1].duration)
			return best[0] == query

	
	retimed_text = []
	for ldc_seg in ldc_tl:
	
		# first cleanup any unattached words that dont fit ldc segment

		while word_times and (word_times[0][0].end <= ldc_seg.start):
				next = word_times[0]
				retimed_text.append([next[0].start, next[0].end, next[1]])
				word_times = word_times[1:]
			
		interval_words = [(tim,txt) for tim,txt in word_times if _ovls(tim, ldc_seg)]

		if interval_words:
			if not is_best_interval(interval_words[0],ldc_seg,ldc_tl):
				# should only happen here if empty ldc segment
				assert len(interval_words) == 1
			
			if not is_best_interval(interval_words[-1],ldc_seg,ldc_tl):
				interval_words = interval_words[:-1]
			
			interval_texts = ' '.join([txt for tim,txt in interval_words])

			if interval_texts:
				word_times = word_times[len(interval_words):]
				retimed_text.append([ldc_seg.start, ldc_seg.end, interval_texts])
			
		
	for i in range(1, len(retimed_text)):
		try:
			assert retimed_text[i][0] > retimed_text[i-1][0]
		except:
			#print(retimed_text[i-1], retimed_text[i])
			assert retimed_text[i][0] >= retimed_text[i-1][0]
			assert retimed_text[i][1] > retimed_text[i-1][1]
			#assert retimed_text[i][1] >= retimed_text[i-1][1]
		
	return retimed_text





# do forced alignment for long audios
# https://github.com/MahmoudAshraf97/ctc-forced-aligner
# (but install the one from https://github.com/catiR/ctc-forced-aligner)
def align_acode(data_files, aln_model = 'is', sad_executable = None):

	assert aln_model.lower() == 'is', ('Model spec must be is [icelandic]')
									 #'or mul [tilingual]')
	# dont use multilingual until uroman package fixed!

	###### load alignment models ######
	language = "isl" # ISO-639-3 Language code
	device = "cuda" if torch.cuda.is_available() else "cpu"
	batch_size = 2 #16
	window_length=30
	merge_threshold = 0.0 # minimum silence duration 150 ms in theory,
	                      #but even at 0 it rarely finds pauses

	if aln_model == 'is':
		model_path = ("language-and-voice-lab/"
						  "wav2vec2-large-xlsr-53-icelandic-ep30-967h")
		romanize = False
		attention_implementation = 'pad'
		context_length = 0.20
		# this model doesnt return attention mask, so pad input only

	elif aln_model == 'mul':
		# this has attention and Star emit, REQUIRES romanize=True
		model_path = "MahmoudAshraf/mms-300m-1130-forced-aligner"
		romanize = True
		context_length = 2.0
		attention_implementation = None #not actually none, uses a default

	alignment_model, alignment_tokenizer = load_alignment_model(
		device,
		dtype=torch.float16 if device == "cuda" else torch.float32,
		attn_implementation = attention_implementation,
		)

	
	for speaker,file_paths in [x for x in data_files['patient'].items()] + \
			[x for x in data_files['control'].items()]:
		if not os.path.exists(file_paths['gold-cfalnj']):

		###### run forced alignment ######
			print(f'MA-ctc-force-aln forced alignment, {speaker}')
			xcp = ctc_prep_transcript(file_paths['xcp-gold'],
										  file_paths['gold-cfanorm'])
			xcp = ' '.join(xcp)

			tokens_starred, text_starred = preprocess_text(xcp,
				romanize=romanize, language=language,
				star_frequency="custom")

			audio_waveform = load_audio(file_paths['wav'],
							alignment_model.dtype, alignment_model.device)
			
			emissions, stride = generate_emissions(alignment_model,
				audio_waveform,	window_length=window_length,
				context_length=context_length, batch_size=batch_size)

			segments, scores, blank_token = get_alignments(emissions,
				tokens_starred, alignment_tokenizer)

			spans = get_spans(tokens_starred, segments, blank_token)
			
			timestamps = postprocess_results(text_starred, spans, stride,
								scores, merge_threshold = merge_threshold)

			with open(file_paths['gold-cfalnj'], "w") as handle:
				json.dump({"text": xcp, "segments": timestamps},
							  handle, indent=4)
			
			
		###### timed diarised transcripts for ELAN ######
			
		if sad_executable:
			wav16 = wav16mono(file_paths['wav'],file_paths['tmp-wav'])
			_ = ldc_sad(wav16, sad_executable, file_paths['ldc-sad-lab'])
			
			_ = label_speakers_alignments(file_paths['xcp-gold'],
				file_paths['gold-cfalnj'], file_paths['gold-cfaspk'], 
				file_paths['ldc-sad-lab'], file_paths['gold-ldcspk'])

		
		else:
			_ = label_speakers_alignments(file_paths['xcp-gold'],
				file_paths['gold-cfalnj'], file_paths['gold-cfaspk'])




if __name__ == "__main__":
	
	original_data_dir = '/home/cati/proj/acode/NextCloud/Data/'
	save_dir = f'../../acoustic-processing/output-is020/'

	sad_executable = '../../acoustic-processing/acodenv/bin/ldc-bpcsad'
	
	data_files = setup_nextcloud_outputs(original_data_dir, save_dir)

	align_acode(data_files, sad_executable = sad_executable)
	
	
