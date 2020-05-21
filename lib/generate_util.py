# Copyright 2019 The Magenta Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# This file has been altered from the original version.

"""Helper functions for generating sounds.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import bezier
import matplotlib.pyplot as plt

from magenta import music as mm
import lib.util as util
import numpy as np
import scipy.io.wavfile as wavfile

MAX_VELOCITY = 127.0

def slerp(p0, p1, t):
  """Spherical linear interpolation."""
  omega = np.arccos(np.dot(
      np.squeeze(p0/np.linalg.norm(p0)), np.squeeze(p1/np.linalg.norm(p1))))
  so = np.sin(omega)
  return np.sin((1.0-t)*omega) / so * p0 + np.sin(t*omega)/so * p1


# def load_midi(midi_path, min_pitch=36, max_pitch=84):
def load_midi(midi_path, min_pitch=24, max_pitch=84):
  """Load midi as a notesequence."""
  midi_path = util.expand_path(midi_path)
  ns = mm.midi_file_to_sequence_proto(midi_path)
  pitches = np.array([n.pitch for n in ns.notes])
  velocities = np.array([n.velocity for n in ns.notes])
  start_times = np.array([n.start_time for n in ns.notes])
  end_times = np.array([n.end_time for n in ns.notes])
  valid = np.logical_and(pitches >= min_pitch, pitches <= max_pitch)
  notes = {'pitches': pitches[valid],
           'velocities': velocities[valid],
           'start_times': start_times[valid],
           'end_times': end_times[valid]}
  return ns, notes


def get_random_instruments(model, total_time, secs_per_instrument=2.0):
  """Get random latent vectors evenly spaced in time."""
  n_instruments = 2
  z_instruments = model.generate_z(n_instruments)
  t_instruments = np.linspace(-.0001, total_time, n_instruments)
  return z_instruments, t_instruments

# USE TO INTERPOLATE 
'''def get_random_instruments(model, total_time, secs_per_instrument=2.0):
  """Get random latent vectors evenly spaced in time."""
  n_instruments = int(total_time / secs_per_instrument)
  z_instruments = model.generate_z(n_instruments)
  t_instruments = np.linspace(-.0001, total_time, n_instruments)
  return z_instruments, t_instruments'''


def get_z_notes(start_times, z_instruments, t_instruments):
  """Get interpolated latent vectors for each note."""
  z_notes = []
  for t in start_times:
    idx = np.searchsorted(t_instruments, t, side='left') - 1
    t_left = t_instruments[idx]
    t_right = t_instruments[idx + 1]
    interp = (t - t_left) / (t_right - t_left)
    z_notes.append(slerp(z_instruments[idx], z_instruments[idx + 1], interp))
    # z_notes.append(slerp(z_instruments[0], z_instruments[0], interp))
  z_notes = np.vstack(z_notes)
  return z_notes
  
def get_envelope(t_note_length,
                 sample_rate,
                 audio_length,
                 attack_percent=1.0,
                 attack_slope=0.5,
                 release_percent=25.0,
                 release_slope=0.5):
  """Create an attack sustain release amplitude envelope."""
  i_attack = int(sample_rate * audio_length * (attack_percent / 100))
  
  i_release = int(sample_rate * audio_length * (release_percent / 100))
  
  i_tot = int(sample_rate * audio_length)
  
  envelope = np.ones(i_tot)
  
  def expand_slope(slope):
    if (slope <= 0.5):
      slope = ((slope * 2) ** 2) / 2
    else:
      slope = 1 - ((((1 - slope) * 2) ** 2) / 2)
    return slope
  
  # Bezier curve attack
  a_slope = expand_slope(float(attack_slope))
  a_curve_point = 1 - a_slope
  a_nodes = np.asfortranarray([[0.0, a_curve_point, 1.0], [0.0, a_slope, 1.0]])
  a_curve = bezier.Curve(a_nodes, degree=2)
  a_samples = np.linspace(0.0, 1.0, i_attack)
  envelope[:i_attack] = a_curve.evaluate_multi(a_samples)[1] 
  
  # Bezier curve release
  release_slope = 1 - release_slope
  r_slope = expand_slope(float(release_slope))
  r_curve_point = r_slope
  r_nodes = np.asfortranarray([[0.0, r_curve_point, 1.0], [1.0, r_slope, 0.0]])
  r_curve = bezier.Curve(r_nodes, degree=2)
  r_samples = np.linspace(0.0, 1.0, i_release)
  envelope[-i_release:i_tot] = r_curve.evaluate_multi(r_samples)[1]
  
  """# Linear attack
  envelope[:i_attack] = np.linspace(0.0, 1.0, i_attack)
  # Linear release
  envelope[-i_release:i_tot] = np.linspace(1.0, 0.0, i_release)"""
  
  envelope = np.expand_dims(envelope, axis=1)
  envelope = np.concatenate((envelope, envelope), axis=1)

# def combine_notes(audio_notes, sample_rate, audio_length, start_times, end_times, velocities):
def combine_notes(audio_notes,
                  sample_rate,
                  audio_length,
                  attack_percent,
                  attack_slope,
                  release_percent,
                  release_slope,
                  start_times,
                  end_times,
                  velocities):

  """Combine audio from multiple notes into a single audio clip.

  Args:
    audio_notes: Array of audio [n_notes, audio_samples].
    start_times: Array of note starts in seconds [n_notes].
    end_times: Array of note ends in seconds [n_notes].
    velocities: Array of velocity values [n_notes].
    sr: Integer, sample rate.

  Returns:
    audio_clip: Array of combined audio clip [audio_samples]
  """
  n_notes = len(audio_notes)
  clip_length = end_times.max() + audio_length
  audio_clip = np.zeros(int(clip_length) * sample_rate)
  
  audio_clip = np.expand_dims(audio_clip, axis=1)
  audio_clip = np.concatenate((audio_clip, audio_clip), axis=1)

  for t_start, t_end, vel, i in zip(
      start_times, end_times, velocities, range(n_notes)):
    # Generate an amplitude envelope
    t_note_length = t_end - t_start
    # envelope = get_envelope(t_note_length, sample_rate, audio_length, attack_percent, attack_slope, release_percent, release_slope)
    envelope = get_envelope(t_note_length,
                            sample_rate,
                            audio_length,
                            attack_percent,
                            attack_slope,
                            release_percent,
                            release_slope)
    length = len(envelope)
    audio_note = audio_notes[i, :length]
    audio_note = audio_notes[i, :length] * envelope 
    # Normalize
    audio_note /= audio_note.max()
    audio_note *= (vel / MAX_VELOCITY)
    # Add to clip buffer
    clip_start = int(t_start * sample_rate)
    clip_end = clip_start + length
    audio_clip[clip_start:clip_end] += audio_note

  # Normalize
  audio_clip /= audio_clip.max()
  audio_clip /= 2.0
  return audio_clip


def save_wav(audio, fname, sample_rate):
  wavfile.write(fname, sample_rate, audio.astype('float32'))
  print('Saved to {}'.format(fname))
