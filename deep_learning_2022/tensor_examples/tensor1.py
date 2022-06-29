#Simple operation to illustrate constant and variable
import tensorflow as tf
x = tf.constant(2)
y = tf.Variable(x+4)
#type(x)
#print(x)
total=x+y
#type(total)
print(total)
