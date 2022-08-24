#example of time distributed components with 2 features, 5 distributed time units and 1 output

import numpy as np
from keras.layers import TimeDistributed
from keras.models import Sequential
from keras.layers import Dense

def reshape_training(vect, dim, sample_size):
    T = len(vect)
    vect_train = np.array([vect]*dim)
    vect_train = np.repeat(vect_train,sample_size)
    vect_train = np.reshape(vect_train, (sample_size,T,dim), order = 'F')
    return (vect_train)


sample_size = 3
x_seed = [1,0.8,0.6,0.4,0.2]
y_seed = [0.9,0.7,0.5,0.3,0.1]
y_train = np.array([[y_seed]*sample_size]).reshape(sample_size,len(y_seed),1)

dim_in = 2
dim_out = 1

x_train = reshape_training(x_seed, dim_in, sample_size)
y_train = reshape_training(y_seed, dim_out, sample_size)


print(x_train)
print(y_train)
print(x_train.shape)
print(y_train.shape)

model=Sequential()
model.add(TimeDistributed(Dense(activation='sigmoid', units=1),
                          input_shape=(None,dim_in)))

## Model compilation
model.compile(loss = 'mse', optimizer = 'rmsprop')

## Model training (less than one minute)
np.random.seed(1337)
model.fit(x_train, y_train, epochs = 1, batch_size = 32)
new_input = np.array([[[1,1],[0.9, 0.9],[0.7, 0.7],[0.5,0.5],[0.3,0.3]],[[1,1],[0.9, 0.9],[0.7, 0.7],[0.5,0.5],[0.3,0.3]]])
print(new_input.shape)
print(model.predict(new_input))
