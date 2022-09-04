#taken from https://github.com/ahstat/deep-learning/blob/master/rnn/1_math_structure_of_rnn.py
import tensorflow
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM, TimeDistributed

def reshape_training(vect_seed, dim, sample_size):
    T = len(vect_seed)
    vect_train = np.array([vect_seed]*dim)
    vect_train = np.repeat(vect_train, sample_size)
    vect_train = np.reshape(vect_train, (sample_size, T, dim), order = 'F')
    # vect_train[0]
    return(vect_train)


sample_size = 256
dim_in = 7
dim_out = 5
nb_units = 13

x_seed = [1,0,0,0,0,0]
y_seed = [0.7,0.6,0.5,0.3,0.8,0.2]

x_train = reshape_training(x_seed, dim_in, sample_size)
y_train = reshape_training(y_seed, dim_out, sample_size)

model=Sequential()
model.add(LSTM(input_shape=(None, dim_in),
                    return_sequences=True,
                    units=nb_units,
                    recurrent_activation='sigmoid',
                    activation='tanh'))
model.add(TimeDistributed(Dense(activation='sigmoid', units=dim_out)))

model.compile(loss = 'mse', optimizer = 'rmsprop')
np.random.seed(1337)

model.fit(x_train, y_train, epochs = 5, batch_size = 32)

#############################
# Understanding the weights #
#############################
wt1 = model.get_weights()
wt2 = model.get_weights()[0].shape
wt3 = model.get_weights()[1].shape
wt4 = model.get_weights()[2].shape
wt5 = model.get_weights()[3].shape
wt6 = model.get_weights()[4].shape
print(wt1)
print(wt2)#wx
print(wt3)#wh
print(wt4)#wb
print(wt5)#wo
print(wt6)#wob

W_x = model.get_weights()[0]
W_h = model.get_weights()[1]
b_h = model.get_weights()[2]
w_y = model.get_weights()[3]
b_y = model.get_weights()[4]

x_1 = [1,1,1,1,2,3,4] # size 7
x_2 = [1,2,3,4,3,2,1]
new_input = [x_1, x_2]

print(model.predict(np.array([new_input])))
