

# =============================================================================================================================================sdfgver%$^#$^&*%^fd321code
import yaml
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf

from tensorflow_tts.inference import AutoConfig
from tensorflow_tts.inference import TFAutoModel
from tensorflow_tts.inference import AutoProcessor

# =============================================================================================================================================sdfgver%$^#$^&*%^fd321code
processor = AutoProcessor.from_pretrained("tensorspeech/tts-fastspeech2-ljspeech-en")

# =============================================================================================================================================sdfgver%$^#$^&*%^fd321code
input_text = "i love you so much."
input_ids = processor.text_to_sequence(input_text)

# =============================================================================================================================================sdfgver%$^#$^&*%^fd321code
fastspeech2 = TFAutoModel.from_pretrained("tensorspeech/tts-fastspeech2-ljspeech-en")

# =============================================================================================================================================sdfgver%$^#$^&*%^fd321markdown
'''
# Save to Pb
'''

# =============================================================================================================================================sdfgver%$^#$^&*%^fd321code
# save model into pb and do inference. Note that signatures should be a tf.function with input_signatures.
tf.saved_model.save(fastspeech2, "./test_saved", signatures=fastspeech2.inference)

# =============================================================================================================================================sdfgver%$^#$^&*%^fd321markdown
'''
# Load and Inference
'''

# =============================================================================================================================================sdfgver%$^#$^&*%^fd321code
fastspeech2 = tf.saved_model.load("./test_saved")

# =============================================================================================================================================sdfgver%$^#$^&*%^fd321code
input_text = "There’s a way to measure the acute emotional intelligence that has never gone out of style."
input_ids = processor.text_to_sequence(input_text)

# =============================================================================================================================================sdfgver%$^#$^&*%^fd321code
mel_before, mel_after, duration_outputs, _, _ = fastspeech2.inference(
    input_ids=tf.expand_dims(tf.convert_to_tensor(input_ids, dtype=tf.int32), 0),
    speaker_ids=tf.convert_to_tensor([0], dtype=tf.int32),
    speed_ratios=tf.convert_to_tensor([1.0], dtype=tf.float32),
    f0_ratios =tf.convert_to_tensor([1.0], dtype=tf.float32),
    energy_ratios =tf.convert_to_tensor([1.0], dtype=tf.float32)
)

# =============================================================================================================================================sdfgver%$^#$^&*%^fd321code
mel_after = tf.reshape(mel_after, [-1, 80]).numpy()
fig = plt.figure(figsize=(10, 8))
ax1 = fig.add_subplot(311)
ax1.set_title(f'Predicted Mel-after-Spectrogram')
im = ax1.imshow(np.rot90(mel_after), aspect='auto', interpolation='none')
fig.colorbar(mappable=im, shrink=0.65, orientation='horizontal', ax=ax1)
plt.show()
plt.close()

# =============================================================================================================================================sdfgver%$^#$^&*%^fd321markdown
'''
# Let inference other input to check dynamic shape
'''

# =============================================================================================================================================sdfgver%$^#$^&*%^fd321code
input_text = "The Commission further recommends that the Secret Service coordinate its planning as closely as possible with all of the Federal agencies from which it receives information."
input_ids = processor.text_to_sequence(input_text)

# =============================================================================================================================================sdfgver%$^#$^&*%^fd321code
mel_before, mel_after, duration_outputs, _, _ = fastspeech2.inference(
    input_ids=tf.expand_dims(tf.convert_to_tensor(input_ids, dtype=tf.int32), 0),
    speaker_ids=tf.convert_to_tensor([0], dtype=tf.int32),
    speed_ratios=tf.convert_to_tensor([1.0], dtype=tf.float32),
    f0_ratios =tf.convert_to_tensor([1.0], dtype=tf.float32),
    energy_ratios =tf.convert_to_tensor([1.0], dtype=tf.float32)
)

# =============================================================================================================================================sdfgver%$^#$^&*%^fd321code
mel_after = tf.reshape(mel_after, [-1, 80]).numpy()
fig = plt.figure(figsize=(10, 8))
ax1 = fig.add_subplot(311)
ax1.set_title(f'Predicted Mel-after-Spectrogram')
im = ax1.imshow(np.rot90(mel_after), aspect='auto', interpolation='none')
fig.colorbar(mappable=im, shrink=0.65, orientation='horizontal', ax=ax1)
plt.show()
plt.close()