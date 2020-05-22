# GANSynth-Stereo

Thank you to Google and the Magenta team for making such an excellent neural network synthesizer! This version of GANSynth has been altered to accept stereo files and a single dataset folder as an input pipeline. I also implemented self-attention whenever the generator or discriminator made a spectrogram with an area of 8192 pixels. This is consistent with the original “Self-Attention Generative Adversarial Networks” paper (Zhang et al., 2018.)

You can use any sample rate as long as all your files are the same length and you don’t exceed 67072 total samples, which is the ceiling for the 128 X 1024 spectrogram implemented in the architecture. 1.5 second samples at 44100 kHz is a good choice. 16-bit only, please. Also, be sure to end all your input files with exactly “.wav” or the algorithm will complain—and no funny characters or spaces in the file names, either.

I tested the algorithm on Colaboratory/Mac, so you may have to do a bit of file-system tweaking for other configurations. And it doesn’t work on Tensorflow 2 yet, so you'll have to include this Python “magic” when you run either the trainer or the generator:

```bash
%tensorflow_version 1.15
```

It’s probably overkill, but I included every possible line to stop the invocation of Tensorflow 2, ending up with this **training** script:

```bash
%tensorflow_version 1.15
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
!pip install bezier
!python gansynth_train.py --hparams='{"train_data_path":"path", "train_root_dir":"path", "channel_mode":"stereo", "audio_length":1.5, "sample_rate":44100}'
```

Note the “!pip install bezier” line. That’s needed for the attack and release envelope curves. The bottommost line is the most important, particularly the “—hparams” section. This is where you tell the algorithm how to train. “train_data_path” is where your input folder lives. “train_root_dir” is where you want your training files to land. “channel_mode” lets you choose either a mono or a stereo model. The program will convert input files to whatever mode you choose here. “audio_length” is the length of your dataset wavs, in seconds. “sample_rate” is self-explanatory.

There are a ton of hyperparameters originally coded by Google, too, including all the standards like learning rate and number of epochs (referred to as “number of images” in the code.) You can alter these either from the command line or in the "model.py" file.

Once you’ve trained a model, it’s time to **generate**. GANSynth uses MIDI files for this purpose, and it can be finicky about what MIDI it likes. If you’re getting errors, try increasing the note length. For a [recent remix](http://soundcloud.com/demonflexcouncil/ymmwina), I ended up just generating single MIDI notes and stitching them together in Ableton. The generation script looks something like this:

```bash
%tensorflow_version 1.15
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
!pip install bezier
!python gansynth_generate.py --ckpt_dir=path --output_dir=path --midi_file=path --attack_percent=1 --attack_slope=0.5 --release_percent=1 --release_slope=0.5
```

You can see that the bottom line looks a little different from the training script. No “—hparams” this time, but the idea is the same—you’re telling the algorithm how you want it to generate audio. "--ckpt_dir" should be the same as "train_data_path" from the training script. "--output_dir" is where you want your generated audio file to go, and "--midi_file" is where your MIDI file lives. Attack and release slopes pertain to the bezier curves--the range is 0 to 1, and higher numbers mean quicker changes. Attack and release percentages (range: 0-100) control fades over the entire length of the note, which means that values above 50 may result in overlapping fades. That’s about it. If you have any questions, don’t hesitate to contact me.



