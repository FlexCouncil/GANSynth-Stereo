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

"""Module contains a registry of dataset classes."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import collections
import struct
import lib.spectral_ops as spectral_ops
import lib.util as util
import numpy as np
import tensorflow as tf
from tensorflow.contrib import lookup as contrib_lookup
from scipy.io.wavfile import read as read_audio

Counter = collections.Counter


class BaseDataset(object):
  """A base class for reading data from disk."""

  def __init__(self, config):
    self._config = config
    self._channel_mode = config['channel_mode']
    self._train_data_path = util.expand_path(config['train_data_path'])

  def provide_one_hot_labels(self, batch_size):
    """Provides one-hot labels."""
    raise NotImplementedError

  def provide_dataset(self):
    """Provides audio dataset."""
    raise NotImplementedError

  def get_pitch_counts(self):
    """Returns a dictionary {pitch value (int): count (int)}."""
    raise NotImplementedError

  '''def get_pitches(self, num_samples):
    """Returns pitch_counter for num_samples for given dataset."""
    all_pitches = []
    pitch_counts = self.get_pitch_counts()
    for k, v in pitch_counts.items():
      all_pitches.extend([k]*v)
    sample_pitches = np.random.choice(all_pitches, num_samples)
    pitch_counter = Counter(sample_pitches)
    return pitch_counter'''


class DatasetFromDirectory(BaseDataset):
  """A dataset for reading NSynth from a TFRecord file."""

  def _get_dataset_from_path(self):
    dataset = tf.data.Dataset.list_files(self._train_data_path)
    dataset = dataset.apply(
        tf.contrib.data.shuffle_and_repeat(buffer_size=1000))
    dataset = dataset.apply(
        tf.contrib.data.parallel_interleave(
            tf.data.TFRecordDataset, cycle_length=20, sloppy=True))
    return dataset

  def provide_one_hot_labels(self, batch_size):
    """Provides one hot labels."""
    pitch_counts = self.get_pitch_counts()
    pitches = sorted(pitch_counts.keys())
    counts = [pitch_counts[p] for p in pitches]
    indices = tf.reshape(
        tf.multinomial(tf.log([tf.to_float(counts)]), batch_size), [batch_size])
    one_hot_labels = tf.one_hot(indices, depth=len(pitches))
    return one_hot_labels

  # NEW
  def provide_dataset(self):
    """Provides dataset (audio, labels) from directory of ".wav" files."""
    audio_length = int(self._config['audio_length'] * self._config['sample_rate'])

    pitch_counts = self.get_pitch_counts()
    pitches = sorted(pitch_counts.keys())
    print('LENGTH OF PITCH ONE HOT')
    print(len(pitches))
    label_index_table = tf.contrib.lookup.index_table_from_tensor(
        sorted(pitches), dtype=tf.int64)
        
    # Get audio file names
    wav_directory = self._config['train_data_path']
    file_names = os.listdir(wav_directory)
    
    # Create seed arrays

    # STEREO
    if (self._channel_mode == 'stereo'):
      wavs = np.empty(shape = [1, audio_length, 2])
      
    # MONO
    if (self._channel_mode == 'mono'):
      wavs = np.empty(shape = [1, audio_length, 1])

    labels = 60

    # Build audio and pitch arrays
    for file_name in file_names:
      # Read audio
      unused_sample_rate, wav = read_audio(wav_directory + '/' + file_name)
      
      print('WAV SHAPE--AFTER READ FROM DIRECTORY, BEFORE LABEL PARSE')
      print(wav.shape)
      
      # Parse MIDI note number of ".wav" file name
      # Follow nsynth file name formatting if MONO, slowcore if STEREO     
      if (wav.ndim == 1):
        label = int(file_name[-11:-8]) - 24
      else:
        label = int(file_name[-7:-4]) - 24
        
      # mono to stereo
      if (self._channel_mode == 'stereo' and wav.ndim == 1):
        wav = np.expand_dims(wav, axis=1)
        wav = np.concatenate((wav, wav), axis=1)

      if (self._channel_mode == 'mono' and wav.ndim == 1):
      # MONO (none, maybe)
        wav = np.expand_dims(wav, axis=1)
        
      # stereo to mono
      if (self._channel_mode == 'mono' and wav.ndim == 2):
        wav = np.mean(wav, axis=1)
        wav = np.expand_dims(wav, axis=1)
    
      # crop
      wav = wav[0:audio_length,:]

      # Put in range [-1, 1]
      float_normalizer = float(np.iinfo(np.int16).max)
      wav = wav / float_normalizer

      # Convert to float32
      wav = np.float32(wav)
      
      # Create batch dimension
      wav = np.expand_dims(wav, axis=0)    

      # Create vertical array of wavs
      
      # STEREO
      if (self._channel_mode == 'stereo'):
        wavs = np.concatenate((wavs, wav), axis=0)   
      
      # MONO
      if (self._channel_mode == 'mono'):
        wavs = np.vstack((wavs, wav))  

      # Create vertical array of pitches
      labels = np.vstack((labels, label))

    # Remove seed arrays
      
    # STEREO
    if (self._channel_mode == 'stereo'):
      wavs = tf.slice(wavs, [1, 0, 0], [len(file_names), audio_length, 2])

    # MONO
    if (self._channel_mode == 'mono'):
      wavs = tf.slice(wavs, [1, 0, 0], [len(file_names), audio_length, 1])
      
    labels = tf.slice(labels, [1, 0], [len(file_names), 1])

    # Translate MIDI note numbers to one-hot vectors
    def _one_hot(wavs, labels):
      labels = tf.one_hot(labels, depth=len(pitches))[0]
      # labels = tf.Print(labels, [labels], message="One hot labels: ")
      labels = tf.dtypes.cast(labels, tf.int64)
      wavs = tf.dtypes.cast(wavs, tf.float32)
      return wavs, labels

    # Create and map dataset
    dataset = tf.data.Dataset.from_tensor_slices((wavs, labels))
    dataset = dataset.map(_one_hot, num_parallel_calls=4)

    return dataset

  def get_pitch_counts(self):
    pitch_counts = {
        24: 711,
        25: 720,
        26: 715,
        27: 725,
        28: 726,
        29: 723,
        30: 738,
        31: 829,
        32: 839,
        33: 840,
        34: 860,
        35: 870,
        36: 999,
        37: 1007,
        38: 1063,
        39: 1070,
        40: 1084,
        41: 1121,
        42: 1134,
        43: 1129,
        44: 1155,
        45: 1149,
        46: 1169,
        47: 1154,
        48: 1432,
        49: 1406,
        50: 1454,
        51: 1432,
        52: 1593,
        53: 1613,
        54: 1578,
        55: 1784,
        56: 1738,
        57: 1756,
        58: 1718,
        59: 1738,
        60: 1789,
        61: 1746,
        62: 1765,
        63: 1748,
        64: 1764,
        65: 1744,
        66: 1677,
        67: 1746,
        68: 1682,
        69: 1705,
        70: 1694,
        71: 1667,
        72: 1695,
        73: 1580,
        74: 1608,
        75: 1546,
        76: 1576,
        77: 1485,
        78: 1408,
        79: 1438,
        80: 1333,
        81: 1369,
        82: 1331,
        83: 1295,
        84: 1291
    }
    return pitch_counts


registry = {
    'dataset_directory': DatasetFromDirectory,
}
