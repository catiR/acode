from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import torch
import numpy as np
import soundfile as sf
from dataclasses import dataclass


def aligner(model_path,model_word_separator = '|', model_blank_token = '[PAD]'):

    # build labels dict from a processor where it is not directly accessible
    def get_processor_labels(processor,word_sep,max_labels=100):
        ixs = sorted(list(range(max_labels)),reverse=True)
        return {processor.tokenizer.decode(n) or word_sep:n for n in ixs}

    #------------------------------------------
    # setup wav2vec2
    #------------------------------------------

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.random.manual_seed(0)
    max_labels = 100 # any reasonable number higher than vocab + extra + special tokens in any language used


    model = Wav2Vec2ForCTC.from_pretrained(model_path).to(device)
    processor = Wav2Vec2Processor.from_pretrained(model_path)
    labels_dict = get_processor_labels(processor,model_word_separator)
    blank_id = labels_dict[model_blank_token]


    #convert frame-numbers to timestamps in seconds
    # w2v2 step size is about 20ms, or 50 frames per second
    def f2s(fr):
        return fr/50



    #------------------------------------------
    # forced alignment with ctc decoder
    #   based on implementation of
    #   https://pytorch.org/audio/main/tutorials/forced_alignment_tutorial.html
    #------------------------------------------


    # return the label class probability of each audio frame
    # wav is the wav data already read in, NOT the file path.
    def get_frame_probs(wav):
        with torch.inference_mode(): # similar to with torch.no_grad():
            input_values = processor(wav,sampling_rate=16000).input_values[0]
            input_values = torch.tensor(input_values, device=device).unsqueeze(0)
            emits = model(input_values).logits
            emits = torch.log_softmax(emits, dim=-1)
        return emits[0].cpu().detach()


    def get_trellis(emission, tokens, blank_id):
    
        num_frame = emission.size(0)
        num_tokens = len(tokens)
        trellis = torch.empty((num_frame + 1, num_tokens + 1))
        trellis[0, 0] = 0
        trellis[1:, 0] = torch.cumsum(emission[:, 0], 0) # len of this slice of trellis is len of audio frames)
        trellis[0, -num_tokens:] = -float("inf") # len of this slice of trellis is len of transcript tokens
        trellis[-num_tokens:, 0] = float("inf")
        for t in range(num_frame):
            trellis[t + 1, 1:] = torch.maximum(
                # Score for staying at the same token
                trellis[t, 1:] + emission[t, blank_id],
                # Score for changing to the next token
                trellis[t, :-1] + emission[t, tokens],
            )
        return trellis



    @dataclass
    class Point:
        token_index: int
        time_index: int
        score: float
    
    @dataclass
    class Segment:
        label: str
        start: int
        end: int
        score: float

        @property
        def mfaform(self):
            return f"{f2s(self.start)},{f2s(self.end)},{self.label}"

        @property
        def length(self):
            return self.end - self.start
    
    
    
    def backtrack(trellis, emission, tokens, blank_id):
    # Note:
    # j and t are indices for trellis, which has extra dimensions
    # for time and tokens at the beginning.
    # When referring to time frame index `T` in trellis,
    # the corresponding index in emission is `T-1`.
    # Similarly, when referring to token index `J` in trellis,
    # the corresponding index in transcript is `J-1`.
        j = trellis.size(1) - 1
        t_start = torch.argmax(trellis[:, j]).item()
    
        path = []
        for t in range(t_start, 0, -1):
        # 1. Figure out if the current position was stay or change
        # `emission[J-1]` is the emission at time frame `J` of trellis dimension.
        # Score for token staying the same from time frame J-1 to T.
            stayed = trellis[t - 1, j] + emission[t - 1, blank_id]
        # Score for token changing from C-1 at T-1 to J at T.
            changed = trellis[t - 1, j - 1] + emission[t - 1, tokens[j - 1]]

        # 2. Store the path with frame-wise probability.
            prob = emission[t - 1, tokens[j - 1] if changed > stayed else 0].exp().item()
        # Return token index and time index in non-trellis coordinate.
            path.append(Point(j - 1, t - 1, prob))
            
            # 3. Update the token
            if changed > stayed:
                j -= 1
                if j == 0:
                    break
        else:
            raise ValueError("Failed to align")
        return path[::-1]


    def merge_repeats(path,transcript):
        i1, i2 = 0, 0
        segments = []
        while i1 < len(path):
            while i2 < len(path) and path[i1].token_index == path[i2].token_index: # while both path steps point to the same token index
                i2 += 1
            score = sum(path[k].score for k in range(i1, i2)) / (i2 - i1)
            segments.append( # when i2 finally switches to a different token,
                Segment(
                    transcript[path[i1].token_index],# to the list of segments, append the token from i1
                    path[i1].time_index, # time of the first path-point of that token
                    path[i2 - 1].time_index + 1, # time of the final path-point for that token.
                    score,
                )
            )
            i1 = i2
        return segments



    def merge_words(segments, separator):
        words = []
        i1, i2 = 0, 0
        while i1 < len(segments):
            if i2 >= len(segments) or segments[i2].label == separator:
                if i1 != i2:
                    segs = segments[i1:i2]
                    word = "".join([seg.label for seg in segs])
                    score = sum(seg.score * seg.length for seg in segs) / sum(seg.length for seg in segs)
                    words.append(Segment(word, segments[i1].start, segments[i2 - 1].end, score))
                i1 = i2 + 1
                i2 = i1
            else:
                i2 += 1
        return words




    #------------------------------------------
    # handle, i/o, etc.
    #------------------------------------------


    # generate mfa format for character (phone) and word alignments
    # skip the word separator as it is not a phone
    def mfalike(chars,wds,wsep):
        hed = ['Begin,End,Label,Type,Speaker\n']
        wlines = [f'{w.mfaform},words,000\n' for w in wds]
        slines = [f'{ch.mfaform},phones,000\n' for ch in chars if ch.label != wsep]
        return (''.join(hed+wlines+slines))

    # generate basic exportable list format for character OR word alignments
    # skip the word separator as it is not a phone
    def basic(segs,wsep="|"):
        return [[s.label,f2s(s.start),f2s(s.end)] for s in segs if s.label != wsep]
        
        
    # generate numbered dicts to use in dtw
    # alignment is given in numbered frames, not converted to timestamps
    def fordtw(words,segments):

        # index i, and word/seg, startframe, endframe
        # preppend the index i to the word or seg
        def _ix(i,elem):
            return [f'{i:03d}__{elem.label}', elem.start, elem.end]
    
        w_al = [_ix(i,wse) for i,wse in enumerate(words)] # from tuple to list

        wsegdict = {}
        for w,s,e in w_al:
            nlett = len(w.split('__')[1])
            wsegs = segments[:nlett]
            wstart = s
            wsegs = [_ix(i,cse) for i,cse in enumerate(wsegs)]
            wsegs = [[seg, ss-s, se-s] for seg,ss,se in wsegs]
            wsegdict[w] = wsegs
            segments = segments[nlett:]
        
        return w_al, wsegdict


    # basic cleaning
    # skip with is_normed=True 
    #  if transcript was already normalised externally
    def normalise_transcript(xcp):
        xcp = xcp.lower()
        xcp = xcp.replace('-','')
        while '  ' in xcp:
            xcp = xcp.replace('  ', ' ')
        return xcp


    # needs pad labels added to correctly time first segment
    # and therefore add word sep character as placeholder in transcript
    def prep_transcript(xcp,is_normed):
        if not is_normed:
            xcp = normalise_transcript(xcp)
        xcp = xcp.replace(' ',model_word_separator)
        label_ids = [labels_dict[c] for c in xcp]
        label_ids = [blank_id] + label_ids + [blank_id]
        xcp = f'{model_word_separator}{xcp}{model_word_separator}'
        return xcp,label_ids
    
    
    def _align(wav_data,transcript,is_normed=False):

        norm_transcript,rec_label_ids = prep_transcript(transcript,is_normed)
        emit = get_frame_probs(wav_data)
        trellis = get_trellis(emit, rec_label_ids, blank_id)
        path = backtrack(trellis, emit, rec_label_ids, blank_id)
        
        segments = merge_repeats(path,norm_transcript)
        words = merge_words(segments, model_word_separator)

        #return fordtw(words,model_word_separator), basic(segments,model_word_separator)
        return basic(words,model_word_separator)

    return _align


# usage:
# from ctcalign import aligner
# model_path ="../models/LVL/wav2vec2-large-xlsr-53-icelandic-ep10-1000h"
# model_word_sep = '|'
# model_blank_tk = '[PAD]'
# caligner = aligner(model_path,model_word_sep,model_blank_tk)
# word_aln, seg_aln = caligner(readwav(wav_path),transcript_string)



