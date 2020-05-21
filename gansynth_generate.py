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

r"""Generate samples with a pretrained GANSynth model.

To use a config of hyperparameters and manual hparams:
>>> python magenta/models/gansynth/generate.py \
>>> --ckpt_dir=/path/to/ckpt/dir --output_dir=/path/to/output/dir \
>>> --midi_file=/path/to/file.mid

If a MIDI file is specified, notes are synthesized with interpolation between
latent vectors in time. If no MIDI file is given, a random batch of notes is
synthesized.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import absl.flags
import lib.flags as lib_flags
import lib.generate_util as gu
import lib.model as lib_model
import lib.util as util
import tensorflow as tf
import numpy as np


absl.flags.DEFINE_string('ckpt_dir',
                         '/tmp/gansynth/acoustic_only',
                         'Path to the base directory of pretrained checkpoints.'
                         'The base directory should contain many '
                         '"stage_000*" subdirectories.')
absl.flags.DEFINE_string('output_dir',
                         '/tmp/gansynth/samples',
                         'Path to directory to save wave files.')
absl.flags.DEFINE_string('midi_file',
                         '',
                         'Path to a MIDI file (.mid) to synthesize.')
absl.flags.DEFINE_integer('batch_size', 8, 'Batch size for generation.')
absl.flags.DEFINE_float('secs_per_instrument', 6.0,
                        'In random interpolations, the seconds it takes to '
                        'interpolate from one instrument to another.')
absl.flags.DEFINE_float('attack_percent', 1.0,
                        'Percentage of note to apply attack envelope.')
absl.flags.DEFINE_float('attack_slope', 0.5,
                        'Slope of attack curve.')
absl.flags.DEFINE_float('release_percent', 25.0,
                        'Percentage of note to apply release envelope.')
absl.flags.DEFINE_float('release_slope', 0.5,
                        'Slope of release curve.')

FLAGS = absl.flags.FLAGS
tf.logging.set_verbosity(tf.logging.INFO)


def main(unused_argv):
  absl.flags.FLAGS.alsologtostderr = True

  # Load the model
  flags = lib_flags.Flags({'batch_size_schedule': [FLAGS.batch_size]})
  model = lib_model.Model.load_from_path(FLAGS.ckpt_dir, flags)
  
  # Get configs
  sample_rate = model.get_sample_rate()
  audio_length = model.get_audio_length()

  # Make an output directory if it doesn't exist
  output_dir = util.expand_path(FLAGS.output_dir)
  if not tf.gfile.Exists(output_dir):
    tf.gfile.MakeDirs(output_dir)

  if FLAGS.midi_file:
    # If a MIDI file is provided, synthesize interpolations across the clip
    unused_ns, notes = gu.load_midi(FLAGS.midi_file)

    # Distribute latent vectors linearly in time
    z_instruments, t_instruments = gu.get_random_instruments(
        model,
        notes['end_times'][-1],
        secs_per_instrument=FLAGS.secs_per_instrument)

    # Get latent vectors for each note
    z_notes = gu.get_z_notes(notes['start_times'], z_instruments, t_instruments)

    # Generate audio for each note
    print('Generating {} samples...'.format(len(z_notes)))
    # Load Instrument
    audio_notes = model.generate_samples_from_z(z_notes, notes['pitches'])

    # Make a single audio clip
    audio_clip = gu.combine_notes(audio_notes,
                                  sample_rate,
                                  audio_length,
                                  FLAGS.attack_percent,
                                  FLAGS.attack_slope,
                                  FLAGS.release_percent,
                                  FLAGS.release_slope,
                                  notes['start_times'],
                                  notes['end_times'],
                                  notes['velocities'])

    # Write the wave files
    fname = os.path.join(output_dir, 'generated_clip.wav')
    gu.save_wav(audio_clip, fname, sample_rate)
    
    # Save z note
    fname = os.path.join(output_dir, 'instrument.npy')
    np.save(fname, z_notes[0])
  else:
    # Otherwise, just generate a batch of random sounds
    waves = model.generate_samples(FLAGS.batch_size)
    # Write the wave files
    for i in range(len(waves)):
      fname = os.path.join(output_dir, 'generated_{}.wav'.format(i))
      gu.save_wav(waves[i], fname, sample_rate)


def console_entry_point():
  tf.app.run(main)


if __name__ == '__main__':
  console_entry_point()
