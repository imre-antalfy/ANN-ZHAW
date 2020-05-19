

"""

Lets do this shit

Might needed pips:
    
    This is for the modelling itself
    pip install music21
    pip install np_utils
    pip install tensorflow==2.1
    
    To speed up the model, i used my GPU
    chronologically:
    pip install python-dev-tools
    pip install tensorflow-gpu
    pip install pycuda

"""


#%% import/dependencies
from music21 import converter, instrument, note, chord
import pathlib
import numpy as np

# keras imports
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import Activation
from keras.layers import BatchNormalization as BatchNorm
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint

#%% Data input
# read midi files

notes = []

p = pathlib.Path("LSTM_Keras/midi_songs")
for file in p.iterdir(): 
    
    # loading each file into a Music21 stream object
    midi = converter.parse(file)
    notes_to_parse = None   
    
    parts = instrument.partitionByInstrument(midi)   
    
    if parts: # file has instrument parts
        notes_to_parse = parts.parts[0].recurse()
    else: # file has notes in a flat structure
        notes_to_parse = midi.flat.notes 
        
    # encode each note with a dot, so encoding in the end is easier
    for element in notes_to_parse:
        if isinstance(element, note.Note):
            notes.append(str(element.pitch))
        # encode each note with a dot, so encoding in the end is easier
        elif isinstance(element, chord.Chord):
            notes.append('.'.join(str(n) for n in element.normalOrder))
            
            
#%% Data preparation 
# Convert categorical to numerical data

sequence_length = 30

# missing in tutorial!!!
# get number of unique notes, done with the set-trick
myset = set(notes)
n_vocab = len(myset)
del(myset)

# get all pitch names
pitchnames = sorted(set(item for item in notes))

# create a dictionary to map pitches to integers
note_to_int = dict((note, number) for number, note in enumerate(pitchnames))

network_input = []
network_output = []

# create input sequences and the corresponding outputs
for i in range(0, len(notes) - sequence_length, 1):
    sequence_in = notes[i:i + sequence_length]
    sequence_out = notes[i + sequence_length]
    network_input.append([note_to_int[char] for char in sequence_in])
    network_output.append(note_to_int[sequence_out])
    
n_patterns = len(network_input)

# reshape the input into a format compatible with LSTM layers
network_input = np.reshape(network_input, (n_patterns, sequence_length, 1))

# normalize input
network_input = network_input / float(n_vocab)

# one-hot encode the output !!! why? define
network_output = to_categorical(network_output)

#%% Model definition, 4 layers
"""
LSTM layers is a Recurrent Neural Net layer that takes a sequence as an input 
and can return either sequences (return_sequences=True) or a matrix.

Dropout layers are a regularisation technique that consists of setting a 
fraction of input units to 0 at each update during the training to prevent 
overfitting. The fraction is determined by the parameter used with the layer.

Dense layers or fully connected layers is a fully connected neural network 
layer where each input node is connected to each output node.

The Activation layer determines what activation function our neural network 
will use to calculate the output of a node.
"""

model = Sequential()
model.add(LSTM(
    256,
    input_shape=(network_input.shape[1], network_input.shape[2]),
    return_sequences=True
))
model.add(Dropout(0.3))
model.add(LSTM(512, return_sequences=True))
model.add(Dropout(0.3))
model.add(LSTM(256))
model.add(Dense(256))
model.add(Dropout(0.3))
model.add(Dense(n_vocab))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

# loss calculation
# based on categorical cross entropy !!!what?
# i imported the file from him
# !!! what is this file???

# explain checkpoints: saving to a specific file, to be able to stop training
# without loosing the weights of the previous runs
filepath = "ANN-ZHAW/LSTM_Keras/weights.hdf5" 
checkpoint = ModelCheckpoint(
    filepath, monitor='loss', 
    verbose=0,        
    save_best_only=True,        
    mode='min'
)    

callbacks_list = [checkpoint]     
model.fit(network_input, 
          network_output, 
          epochs=200, 
          batch_size=64, 
          callbacks=callbacks_list)


























