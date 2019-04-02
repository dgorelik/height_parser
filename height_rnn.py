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

TRAINING_LOOPS = 1000
MIN_INCHES = 20
MAX_INCHES = 90
train_x = []
train_y = []
for i in range(TRAINING_LOOPS):
  for inches in range(MIN_INCHES, MAX_INCHES):
    train_x.append(format_inches(inches))
    train_x.append(format_feet(inches))
    train_y.extend([inches, inches])

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
    train_x_int, value=0, padding='post', maxlen=30)
    
# Normalize labels to a range of 0 to 1
train_labels = np.array(train_y) / 100

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