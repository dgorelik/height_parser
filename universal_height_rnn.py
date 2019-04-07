#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import random

def format_inches(inches):
  """Returns an example user-input inches string."""
  if inches == 0:
    return ""
  suffixes = ["in", "inch", "inchs", "inches", "''", "″", "\""]
  suffix = random.choice(suffixes)
  spacing = random.randrange(3)*" "
  return f"{inches}{spacing}{suffix}"

def format_feet(inches):
  """Returns an example user-input feet string."""
  feet = inches // 12
  inches_part =  format_inches(inches % 12)
  suffixes = ["ft", "feet", "fts", "'", "′", ""]
  suffix = random.choice(suffixes)
  spacing_1 = random.randrange(3)*" "
  spacing_2 = random.randrange(3)*" "
  return f"{feet}{spacing_1}{suffix}{spacing_2}{inches_part}"

def format_cm(cm):
  """Returns an example user-input cm string."""
  if cm == 0:
    return ""
  suffixes = ["cm", "centimeters", "centimetres", "centis", "cms", ""]
  suffix = random.choice(suffixes)
  spacing = random.randrange(3)*" "
  return f"{cm}{spacing}{suffix}"
  
def format_meters(cm):
  """Returns an example user-input meters string."""
  if cm < 100:
    return format_cm(100)
  m = cm // 100
  cm_part =  format_cm(cm % 100)
  suffixes = ["meters", "metres", "m", "ms"]
  suffix = random.choice(suffixes)
  spacing_1 = random.randrange(3)*" "
  spacing_2 = random.randrange(3)*" "
  return f"{m}{spacing_1}{suffix}{spacing_2}{cm_part}"

TRAINING_LOOPS = 10000
train_x = []
train_y = []
for i in range(TRAINING_LOOPS):
  cm = np.random.normal(169, 7.42*2*2)
  inches_rounded = int(np.round(cm / 2.54))
  cm_rounded = int(np.round(cm))
  train_x.append(format_inches(inches_rounded))
  train_x.append(format_feet(inches_rounded))
  train_x.append(format_cm(cm_rounded))
  train_x.append(format_meters(cm_rounded))
  
  # Training labels are height in cm, and 0/1 depending on unit
  train_y.extend([[cm, 0], [cm, 0], [cm, 1], [cm, 1]])

maxlen = max([len(inp) for inp in train_x])

# Get entire input as a string
text = ' '.join(train_x)
# The unique chars in our model
vocab = sorted(set(text))

# Make a map of chars to numeric indices
idx_for_char = {u:i for i, u in enumerate(vocab)}
def text_to_idx(text):
  return [idx_for_char[c] for c in text]

train_x_int = [text_to_idx(el) for el in train_x]

train_data = tf.keras.preprocessing.sequence.pad_sequences(
    train_x_int, value=0, padding='post', maxlen=maxlen)
    
# Normalize labels to a range of 0 to 1
norm_factor = max(train_y)[0] * 1.03
train_labels = np.array([[height/norm_factor, units]
                         for height, units in train_y])

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(len(vocab), 64),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid'),
])

model.compile(loss='mse', optimizer='adam', metrics=['mae'])

history = model.fit(
    train_data,
    train_labels,
    epochs=10,
    batch_size=512,
    validation_split=0.2)

def predict(model, height_txt):
  # First, pad as was done for training data
  encoded_text = tf.keras.preprocessing.sequence.pad_sequences(
      [text_to_idx(height_txt)], value=0, padding='post', maxlen=30)
  
  pred = model.predict(encoded_text)
  amount, units = np.squeeze(pred).tolist()
  
  # Training labels were divided by 100, so we mulitiply by 100
  cm = np.squeeze(amount*norm_factor)
  if units < 0.5:
    inches = cm / 2.54
    ft = inches // 12
    inches = np.round(inches % 12)
    return f"{ft} ft {inches} inches ({units})"
  cm = np.round(cm)                
  return f"{cm} cm  ({units})"