import soundfile as sf
import glob,os
from scipy import signal
import numpy as np
import torch, torchaudio
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from faster_whisper import WhisperModel
from pydub import AudioSegment
from scripts.ctcalign import aligner
from scripts.util import *



# handle asrs
def asr_one(wav_file, seg_file, save_prefix,asr_name,asr_spec,asr_ext=None):
	asr_path = asr_spec['path']
	asr_function = eval(asr_spec['recogniser'])
	
	if not asr_ext:
		asr_ext = asr_name
	save_file = f'{save_prefix}{asr_ext}.txt'
	
	if os.path.exists(save_file):
		print(f'Specified ASR {asr_name} already existed, did not rerun.')
	else:
		asr_function(wav_file,seg_file,save_file,asr_path)

	return save_file



# use w2v2 to recognise words in audio segments
#   based on a pre-existing segmentation
def recognise_w2v2(wav_file,seg_file, save_file,w2v2path):

	torch.random.manual_seed(0)
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	w2v2model= Wav2Vec2ForCTC.from_pretrained(w2v2path,use_safetensors=True).to(device)
	w2v2processor = Wav2Vec2Processor.from_pretrained(w2v2path,use_safetensors=True)

	sr = 16000
	whole_audio = readwav(wav_file)
	segments = get_transcript_segments(seg_file)
	
	for i,seg in enumerate(segments):
	
		audio_seg = whole_audio[int(seg['s']*sr):int(seg['e']*sr)]
		try:
			with torch.inference_mode():
				input_values = w2v2processor(audio_seg,sampling_rate=16000).input_values[0]
				input_values = torch.tensor(input_values, device=device).unsqueeze(0)
				logits = w2v2model(input_values).logits
				pred_ids = torch.argmax(logits, dim=-1)
				dec = w2v2processor.batch_decode(pred_ids)
				xcp = dec[0]
		
			segments[i] = {'l':seg['l'], 's':seg['s'], 'e':seg['e'], 't':xcp}
			
		except RuntimeError:
			print(f'Skipping segment {seg["s"]} - {seg["e"]}')
	
	segments = [f'{seg["l"]}\t{seg["s"]}\t{seg["e"]}\t{seg["t"]}' for seg in segments]
	with open(save_file,'w') as handle:
		handle.write('\n'.join(segments))
		


# use fasterwhisper to recognise words in audio segments
#   based on a pre-existing segmentation
def recognise_fasterwhisper(wav_file,seg_file,save_file,whisperpath):

	wdevice = "cuda" if torch.cuda.is_available() else "cpu"
	whispermodel = WhisperModel(model_size_or_path=whisperpath, device=wdevice, local_files_only=True)
	
	sr = 16000
	whole_audio = readwav(wav_file)
	segments = get_transcript_segments(seg_file)
	
	# weird fasterwhisper segment format
	cts = [t for ts in [[seg['s'],seg['e']] for seg in segments] for t in ts]
	assert len(cts) == 2*len(segments)
	print(f'Audio {round(len(whole_audio)/sr)} seconds')
	
	xcps, info = whispermodel.transcribe(audio = whole_audio, language = "is", temperature = [0.03, 0.2, 0.6], clip_timestamps = cts) #no_repeat_ngram_size = 5, [0.03, 0.1, 0.2, 0.4, 0.6],
	
	
	# heuristic to quick cleanup whisper transcriptions.....
	def _cl(output,st,et):
		if (et-st < 0.3) and output in ('það er', 'það er það'):
			return ''
		else:
			return output
	
	whisper_segs = []
	for xcp in xcps:
		xid, xs, xe, xt, xtmp = xcp.id, xcp.start, xcp.end, xcp.text, xcp.temperature
		print(xs, xe, f'({xtmp})', xt, sep='\t')
		whisper_segs.append([xid,xs,xe,_cl(xt,xs,xe)])
		
	for i,seg in enumerate(segments):
		wh_matches = [xt for xid,xs,xe,xt in whisper_segs if round(xs,2)>=round(seg['s'],2) and round(xe,2)<=round(seg['e'],2)]
		if len(wh_matches) == 0:
			print(f'No words recognised in segment {seg["s"]} - {seg["e"]}')
		else:
			wh_matches = ' '.join(wh_matches)
			segments[i] = {'l':seg['l'], 's':seg['s'], 'e':seg['e'], 't':wh_matches}
		
	segments = [f'{seg["l"]}\t{seg["s"]}\t{seg["e"]}\t{seg["t"]}' for seg in segments]
	with open(save_file,'w') as handle:
		handle.write('\n'.join(segments))



# combine following segments from a single speaker
# if time between is less than minimum pause
def cleanup_segments(segments, min_pause = 0.15):
	segments = sorted(segments, key = lambda x: x[0])
	
	clean = [segments[0]]
	et = clean[-1][1]
	for s,e,txt in segments[1:]:
	
		assert s>=et
		if s-et >= min_pause:
			clean.append([s,e,txt])
		else:
			st = clean[-1][0]
			txt2 = f'{clean[-1][2]} {txt}'
			txt2 = txt2.replace('  ', ' ')
			clean[-1] = [st,e,txt2]
		et = e
	return clean



def realign_w2v2(wav_file,firstpass_file,w2v2_aligner):

	secondpass_file = firstpass_file.replace('/asr-1pass/','/asr-2pass/').rsplit('.',1)[0]+'++w2v2F.txt'
	
	if os.path.exists(secondpass_file):
		print(f'Second pass already done: {os.path.basename(secondpass_file)}')
		return secondpass_file
	
	print('Second pass realignment...')
	wav = readwav(wav_file)
	
	with open(firstpass_file,'r') as handle:
		transcript = handle.read().splitlines()
	transcript = [l.split('\t') for l in transcript]
	speakers = set([l[0] for l in transcript])
	
	
	final_segments = []
	
	for sk in speakers:
		old_segments = [l[1:] for l in transcript if l[0]==sk]
		new_segments = []
		for s,e,txt in old_segments:
			words = [w for w in txt.split(' ') if w]
			if len(words) <= 1: #nothing to realign
				new_segments.append([float(s),float(e),txt])
			else:
				six = max(0,int((float(s)*16000)-320))
				eix = min(len(wav),int((float(e)*16000)+320))
				offset = float(s)
				audio_segment = wav[six:eix]
				try:
					word_aln = w2v2_aligner(audio_segment, txt)
					for w, ws, we in word_aln:
						new_segments.append([ws+offset,we+offset,w])
				except:
					print(f'w2v2 failed to align {txt}, {s} - {e}')
					new_segments.append([float(s),float(e),txt])
		new_segments = cleanup_segments(new_segments)
		
		for s,e,txt in new_segments:
			final_segments.append([sk,s,e,txt])
			
	final_segments = sorted(final_segments, key=lambda x: x[1])
	final_segments = [f'{sk}\t{st}\t{et}\t{txt}' for sk,st,et,txt in final_segments]
	
	with open(secondpass_file,'w') as handle:
		handle.write('\n'.join(final_segments))
	
	return secondpass_file
		





