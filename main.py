"""
CLaMIDIa

This module provides functionality for working with MIDI files and encoding/decoding MIDI data using the CLaMIDIa model.

The module includes the following classes and functions:

Variables:
- pos_resolution: The resolution of positions per beat (quarter note).
- bar_max: The maximum number of bars.
- velocity_quant: The quantization value for velocity.
- tempo_quant: The quantization value for tempo.
- min_tempo: The minimum tempo value.
- max_tempo: The maximum tempo value.
- duration_max: The maximum duration value.
- max_ts_denominator: The maximum denominator for time signature.
- max_notes_per_bar: The maximum number of notes per bar.
- beat_note_factor: The factor to convert beats to notes in MIDI format.
- deduplicate: A flag indicating whether to deduplicate notes.
- filter_symbolic: A flag indicating whether to filter symbolic music.
- filter_symbolic_ppl: The threshold perplexity for symbolic music filtering.
- trunc_pos: The truncation value for positions.
- sample_len_max: The maximum length of a sample.
- sample_overlap_rate: The overlap rate for samples.
- ts_filter: A flag indicating whether to apply time signature filtering.
- pool_num: The number of processes in the multiprocessing pool.
- max_inst: The maximum number of instruments.
- max_pitch: The maximum pitch value.
- max_velocity: The maximum velocity value.
- ts_dict: A dictionary mapping time signature tuples to their encoding indices.
- ts_list: A list of time signature tuples.
- dur_enc: A list of duration encodings.
- dur_dec: A list of duration decodings.
- logger: The logger object.
- disable_cp: A flag indicating whether checkpoint is disabled.
- mask_strategy: The mask strategy for encoding.
- convert_encoding: The encoding format to convert.
- crop_length: The length to crop the compound tokens.
- max_bars: The maximum number of bars.
- max_instruments: The maximum number of instruments.
- emb_dict: A dictionary mapping token indices to the number of tokens used to represent each feature.

"""

# CLaMIDIa time
import io
import miditoolkit
import torch.nn as nn
import math
import logging
import os
import torch

from muzic.musicbert.musicbert import *
from model import Clamidia

from typing import Tuple, List

pos_resolution = 16  # per beat (quarter note)
bar_max = 256
velocity_quant = 4
tempo_quant = 12  # 2 ** (1 / 12)
min_tempo = 16
max_tempo = 256
duration_max = 8  # 2 ** 8 * beat
max_ts_denominator = 6  # x/1 x/2 x/4 ... x/64
max_notes_per_bar = 2  # 1/64 ... 128/64
beat_note_factor = 4  # In MIDI format a note is always 4 beats
deduplicate = True
filter_symbolic = False
filter_symbolic_ppl = 16
trunc_pos = 2 ** 16  # approx 30 minutes (1024 measures)
sample_len_max = 1000  # window length max
sample_overlap_rate = 4
ts_filter = False
pool_num = 24
max_inst = 127
max_pitch = 127
max_velocity = 127

ts_dict = dict()
ts_list = list()
for i in range(0, max_ts_denominator + 1):  # 1 ~ 64
    for j in range(1, ((2 ** i) * max_notes_per_bar) + 1):
        ts_dict[(j, 2 ** i)] = len(ts_dict)
        ts_list.append((j, 2 ** i))
dur_enc = list()
dur_dec = list()
for i in range(duration_max):
    for j in range(pos_resolution):
        dur_dec.append(len(dur_enc))
        for k in range(2 ** i):
            dur_enc.append(len(dur_dec) - 1)

logger = logging.getLogger(__name__)
disable_cp = 'disable_cp' in os.environ
print('disable_cp =', disable_cp)
mask_strategy = os.environ['mask_strategy'].split(
    '+') if 'mask_strategy' in os.environ else ['bar']
print('mask_strategy =', mask_strategy)
assert all(item in ['element', 'compound', 'bar'] for item in mask_strategy)
convert_encoding = os.environ['convert_encoding'] if 'convert_encoding' in os.environ else 'OCTMIDI'
print('convert_encoding =', convert_encoding)
crop_length = int(os.environ['crop_length']
                  ) if 'crop_length' in os.environ else None
print('crop_length =', crop_length)  # of compound tokens
max_bars = 256
max_instruments = 256


def t2e(x: Tuple[int, int]):
    """
    Converts a time signature to its corresponding encoding.

    Args:
        x: A tuple representing a time signature.

    Returns:
        The encoding of the time signature.
    """
    assert x in ts_dict, 'unsupported time signature: ' + str(x)
    return ts_dict[x]


def e2t(x: int) -> Tuple[int, int]:
    """
    Converts an encoding to its corresponding time signature.

    Args:
        x: An integer representing an encoding.

    Returns:
        The time signature corresponding to the encoding.
    """
    return ts_list[x]


def d2e(x: int) -> int:
    """
    Converts a duration to its corresponding encoding.

    Args:
        x: An integer representing a duration.

    Returns:
        The encoding of the duration.
    """
    return dur_enc[x] if x < len(dur_enc) else dur_enc[-1]


def e2d(x: int) -> int:
    """
    Converts an encoding to its corresponding duration.

    Args:
        x: An integer representing an encoding.

    Returns:
        The duration corresponding to the encoding.
    """
    return dur_dec[x] if x < len(dur_dec) else dur_dec[-1]


def v2e(x: int) -> int:
    """
    Converts a velocity to its corresponding encoding.

    Args:
        x: An integer representing a velocity.

    Returns:
        The encoding of the velocity.
    """
    return x // velocity_quant


def e2v(x: int) -> int:
    """
    Converts an encoding to its corresponding velocity.

    Args:
        x: An integer representing an encoding.

    Returns:
        The velocity corresponding to the encoding.
    """
    return (x * velocity_quant) + (velocity_quant // 2)


def b2e(x: float) -> int:
    """
    Converts a tempo to its corresponding encoding.

    Args:
        x: A float representing a tempo.

    Returns:
        The encoding of the tempo.
    """
    x = max(x, min_tempo)
    x = min(x, max_tempo)
    x = x / min_tempo
    e = round(math.log2(x) * tempo_quant)
    return e


def e2b(x: int) -> float:
    """
    Converts an encoding to its corresponding tempo.

    Args:
        x: An integer representing an encoding.

    Returns:
        The tempo corresponding to the encoding.
    """
    return 2 ** (x / tempo_quant) * min_tempo


def time_signature_reduce(numerator: int, denominator: int) -> Tuple[int, int]:
    """
    Reduces a time signature to fit within the maximum denominator and maximum notes per bar.

    Args:
        numerator: The numerator of the time signature.
        denominator: The denominator of the time signature.

    Returns:
        The reduced time signature as a tuple.
    """
    # reduction (when denominator is too large)
    while denominator > 2 ** max_ts_denominator and denominator % 2 == 0 and numerator % 2 == 0:
        denominator //= 2
        numerator //= 2
    # decomposition (when length of a bar exceed max_notes_per_bar)
    while numerator > max_notes_per_bar * denominator:
        for i in range(2, numerator + 1):
            if numerator % i == 0:
                numerator //= i
                break
    return numerator, denominator


def MIDI_to_encoding(midi_obj: miditoolkit.midi.parser.MidiFile) -> List[Tuple[int]]:
    """Converts a MIDI file to an encoding.

    This function takes a MIDI file object and converts it to an encoding that
    represents the music in the file. The encoding is a list of 8-tuples, where
    each tuple represents a note and its attributes.

    Args:
        midi_obj: The MIDI file object to convert.

    Returns:
        A list of 8-tuples, where each tuple represents a note and its attributes.
        The elements of the tuple are as follows:
        0: The bar in which the note occurs.
        1: The position of the note within the bar.
        2: The program number of the instrument that plays the note.
        3: The pitch of the note.
        4: The duration of the note.
        5: The velocity of the note.
        6: The time signature in effect at the start of the note.
        7: The tempo in effect at the start of the note.

    Raises:
        AssertionError: If the time signature changes in an invalid way.
    """
    def time_to_pos(t):
        return round(t * pos_resolution / midi_obj.ticks_per_beat)

    notes_start_pos = [time_to_pos(j.start)
                       for i in midi_obj.instruments for j in i.notes]

    if len(notes_start_pos) == 0:
        return list()

    max_pos = min(max(notes_start_pos) + 1, trunc_pos)
    # (Measure, TimeSig, Pos, Tempo)
    pos_to_info = [[None for _ in range(4)] for _ in range(max_pos)]
    tsc = midi_obj.time_signature_changes
    tpc = midi_obj.tempo_changes
    for i in range(len(tsc)):
        for j in range(time_to_pos(tsc[i].time), time_to_pos(tsc[i + 1].time) if i < len(tsc) - 1 else max_pos):
            if j < len(pos_to_info):
                pos_to_info[j][1] = t2e(time_signature_reduce(
                    tsc[i].numerator, tsc[i].denominator))
    for i in range(len(tpc)):
        for j in range(time_to_pos(tpc[i].time), time_to_pos(tpc[i + 1].time) if i < len(tpc) - 1 else max_pos):
            if j < len(pos_to_info):
                pos_to_info[j][3] = b2e(tpc[i].tempo)
    for j in range(len(pos_to_info)):
        if pos_to_info[j][1] is None:
            # MIDI default time signature
            pos_to_info[j][1] = t2e(time_signature_reduce(4, 4))
        if pos_to_info[j][3] is None:
            pos_to_info[j][3] = b2e(120.0)  # MIDI default tempo (BPM)
    cnt = 0
    bar = 0
    measure_length = None
    for j in range(len(pos_to_info)):
        ts = e2t(pos_to_info[j][1])
        if cnt == 0:
            measure_length = ts[0] * beat_note_factor * pos_resolution // ts[1]
        pos_to_info[j][0] = bar
        pos_to_info[j][2] = cnt
        cnt += 1
        if cnt >= measure_length:
            assert cnt == measure_length, 'invalid time signature change: pos = {}'.format(
                j)
            cnt -= measure_length
            bar += 1
    encoding = []
    start_distribution = [0] * pos_resolution
    for inst in midi_obj.instruments:
        for note in inst.notes:
            if time_to_pos(note.start) >= trunc_pos:
                continue
            start_distribution[time_to_pos(note.start) % pos_resolution] += 1
            info = pos_to_info[time_to_pos(note.start)]
            encoding.append((info[0], info[2], max_inst + 1 if inst.is_drum else inst.program, note.pitch + max_pitch +
                             1 if inst.is_drum else note.pitch, d2e(time_to_pos(note.end) - time_to_pos(note.start)), v2e(note.velocity), info[1], info[3]))
    if len(encoding) == 0:
        return list()
    tot = sum(start_distribution)
    start_ppl = 2 ** sum((0 if x == 0 else -(x / tot) *
                          math.log2((x / tot)) for x in start_distribution))
    # filter unaligned music
    if filter_symbolic:
        assert start_ppl <= filter_symbolic_ppl, 'filtered out by the symbolic filter: ppl = {:.2f}'.format(
            start_ppl)
    encoding.sort()
    
    return encoding


def encoding_to_MIDI(encoding):
    """Converts an encoding to a MIDI file.

    This function takes an encoding that represents a piece of music and converts
    it to a MIDI file object. The encoding is a list of 8-tuples, where each tuple
    represents a note and its attributes.

    Args:
        encoding: A list of 8-tuples, where each tuple represents a note and its attributes.
        The elements of the tuple are as follows:
        0: The bar in which the note occurs.
        1: The position of the note within the bar.
        2: The program number of the instrument that plays the note.
        3: The pitch of the note.
        4: The duration of the note.
        5: The velocity of the note.
        6: The time signature in effect at the start of the note.
        7: The tempo in effect at the start of the note.

    Returns:
        A MIDI file object representing the music described by the encoding.

    Raises:
        AssertionError: If the time signature changes in an invalid way.
    """
    # TODO: filter out non-valid notes and error handling
    bar_to_timesig = [list()
                      for _ in range(max(map(lambda x: x[0], encoding)) + 1)]
    for i in encoding:
        bar_to_timesig[i[0]].append(i[6])
    bar_to_timesig = [max(set(i), key=i.count) if len(
        i) > 0 else None for i in bar_to_timesig]
    for i in range(len(bar_to_timesig)):
        if bar_to_timesig[i] is None:
            bar_to_timesig[i] = t2e(time_signature_reduce(
                4, 4)) if i == 0 else bar_to_timesig[i - 1]
    bar_to_pos = [None] * len(bar_to_timesig)
    cur_pos = 0
    for i in range(len(bar_to_pos)):
        bar_to_pos[i] = cur_pos
        ts = e2t(bar_to_timesig[i])
        measure_length = ts[0] * beat_note_factor * pos_resolution // ts[1]
        cur_pos += measure_length
    pos_to_tempo = [list() for _ in range(
        cur_pos + max(map(lambda x: x[1], encoding)))]
    for i in encoding:
        pos_to_tempo[bar_to_pos[i[0]] + i[1]].append(i[7])
    pos_to_tempo = [round(sum(i) / len(i)) if len(i) >
                    0 else None for i in pos_to_tempo]
    for i in range(len(pos_to_tempo)):
        if pos_to_tempo[i] is None:
            pos_to_tempo[i] = b2e(120.0) if i == 0 else pos_to_tempo[i - 1]
    midi_obj = miditoolkit.midi.parser.MidiFile()

    def get_tick(bar: int, pos: int) -> int:
        return (bar_to_pos[bar] + pos) * midi_obj.ticks_per_beat // pos_resolution

    midi_obj.instruments = [miditoolkit.containers.Instrument(program=(
        0 if i == 128 else i), is_drum=(i == 128), name=str(i)) for i in range(128 + 1)]
    for i in encoding:
        start = get_tick(i[0], i[1])
        program = i[2]
        pitch = (i[3] - 128 if program == 128 else i[3])
        duration = get_tick(0, e2d(i[4]))
        if duration == 0:
            duration = 1
        end = start + duration
        velocity = e2v(i[5])
        midi_obj.instruments[program].notes.append(miditoolkit.containers.Note(
            start=start, end=end, pitch=pitch, velocity=velocity))
    midi_obj.instruments = [
        i for i in midi_obj.instruments if len(i.notes) > 0]
    cur_ts = None
    for i in range(len(bar_to_timesig)):
        new_ts = bar_to_timesig[i]
        if new_ts != cur_ts:
            numerator, denominator = e2t(new_ts)
            midi_obj.time_signature_changes.append(miditoolkit.containers.TimeSignature(
                numerator=numerator, denominator=denominator, time=get_tick(i, 0)))
            cur_ts = new_ts
    cur_tp = None
    for i in range(len(pos_to_tempo)):
        new_tp = pos_to_tempo[i]
        if new_tp != cur_tp:
            tempo = e2b(new_tp)
            midi_obj.tempo_changes.append(
                miditoolkit.containers.TempoChange(tempo=tempo, time=get_tick(0, i)))
            cur_tp = new_tp
    return midi_obj

def encoding_to_str(e):
    bar_index_offset = 0
    p = 0
    tokens_per_note = 8
    return ' '.join((['<s>'] * tokens_per_note)
                    + ['<{}-{}>'.format(j, k if j > 0 else k + bar_index_offset) for i in e[p: p +
                                                                                            sample_len_max] if i[0] + bar_index_offset < bar_max for j, k in enumerate(i)]
                    + (['</s>'] * (tokens_per_note
                                   - 1)))   # 8 - 1 for append_eos functionality of binarizer in fairseq

# emb_dict = {0:256, 1:128, 2:129, 3:256, 4:128, 5:32, 6:254, 7:49} - from the paper, number of tokens used to represent each feature
emb_dict = {0:4, 1:260, 2:388, 3:517, 4:773, 5:901, 6:933, 7:1187}

def emb(oct_str, emb_dict=emb_dict, emb_dim=768):
    # oct_str: output from encoding to string
    res = []
    oct_inputs = []
    tokens = oct_str.split()

    for token in tokens: # TODO: use 8-sized window
        if token == '<s>':
            oct_inputs.append(0)
        elif token == '<pad>':
            oct_inputs.append(1)    
        elif token == '</s>':
            oct_inputs.append(2)
        elif token == '<unk>':
            oct_inputs.append(3)    
        else:
            key, val = token[1:-1].split('-')
            oct_inputs.append(emb_dict[int(key)]+int(val))
            
    oct_inputs = torch.IntTensor(oct_inputs)
    for inp in oct_inputs:
        embedding = []
        embed = nn.Embedding(1236, emb_dim)
        for i in range(len(inp)):
            x = embed(inp[i])
            embedding.append(x)
        res.append(embedding)

def gen_dictionary(file_name):
    num = 0
    if os.path.exists(file_name):
        return
    with open(file_name, 'w') as f:
        for j in range(bar_max):
            print('<0-{}>'.format(j), num, file=f)
        for j in range(beat_note_factor * max_notes_per_bar * pos_resolution):
            print('<1-{}>'.format(j), num, file=f)
        for j in range(max_inst + 1 + 1):
            # max_inst + 1 for percussion
            print('<2-{}>'.format(j), num, file=f)
        for j in range(2 * max_pitch + 1 + 1):
            # max_pitch + 1 ~ 2 * max_pitch + 1 for percussion
            print('<3-{}>'.format(j), num, file=f)
        for j in range(duration_max * pos_resolution):
            print('<4-{}>'.format(j), num, file=f)
        for j in range(v2e(max_velocity) + 1):
            print('<5-{}>'.format(j), num, file=f)
        for j in range(len(ts_list)):
            print('<6-{}>'.format(j), num, file=f)
        for j in range(b2e(max_tempo) + 1):
            print('<7-{}>'.format(j), num, file=f)

if __name__ == '__main__':

    # (0 Bar, 1 Pos, 2 Program, 3 Pitch, 4 Duration, 5 Velocity, 6 TimeSig, 7 Tempo)
    ROOT = './input0'
    gen_dictionary('{}/dict.txt'.format(ROOT))
    ROOT = './label'
    gen_dictionary('{}/dict.txt'.format(ROOT))

    num_tokens = [256, 128, 129, 256, 128, 32, 254, 49]
    
    # model = Clamidia(
    #     d_model=768, nhead=12, dropout=0.1,
    #     num_layers=12, dim_feedforward=2048,
    #     num_tokens=num_tokens, activation='gelu'
    # )
    
    filename = 'model/examples/dq.mid'
    with open(filename, 'rb') as f:
            midi_file = io.BytesIO(f.read())
    midi_obj = miditoolkit.midi.parser.MidiFile(file=midi_file)
    enc = encoding_to_str(MIDI_to_encoding(midi_obj))

    roberta_base = MusicBERTModel.from_pretrained('.', 
    checkpoint_file = './checkpoint_last_musicbert_small_w_genre_head.pt',
    user_dir='muzic/musicbert'    # activate the MusicBERT plugin with this keyword
    )
    # ^this seems to be returning a GeneratorHubInterface instead of RobertaHubInterface. If so, we're fucked lmao

    hiddens = roberta_base.extract_features(enc, return_all_hiddens=True) # get outputs of all layers in a tensor, index to find TF output


    