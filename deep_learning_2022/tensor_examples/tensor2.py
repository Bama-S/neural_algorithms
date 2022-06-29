#decode an image
import tensorflow as tf
image = "rose.jpg"
image_open = open(image,'rb')
read_image = image_open.read()
image_decode = tf.image.decode_jpeg(read_image)
print(image_decode)
print(image_decode.shape)
