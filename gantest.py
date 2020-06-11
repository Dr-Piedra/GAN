import tensorflow as tf
from tensorflow import keras as k
import tensorflow_datasets as tfds
from keras.layers import Dense, Dropout
import numpy as np

(X,Y_),(x,y_)=k.datasets.mnist.load_data()
shape = (28,28,1)
img_width, img_height, img_channels = 28,28,1

X=X.reshape(60000,28,28,1)
Y_=k.utils.to_categorical(Y_,10)
#def discriminator(shape = (28,28,1))
#	model.Sequential(

Image_input =  k.layers.Input(shape=[img_width, img_height, img_channels])
L1 =  k.layers.Conv2D(64,kernel_size=(3,3), strides = (2,2), padding='same', activation ="relu")(Image_input)
L2 =  k.layers.Conv2D(64,kernel_size=(3,3), strides = (2,2), padding='same', activation ="relu")(L1)
L3 =  k.layers.Conv2D(64,kernel_size=(3,3), strides = (2,2), padding='same', activation ="relu")(L2)
L4 =  k.layers.Flatten()(L3)
End = k.layers.Dense(2, activation='softmax')(L4)


discriminator = k.models.Model(inputs = [Image_input], outputs = [End])
discriminator.compile(loss = "categorical_crossentropy", optimizer = "adam", metrics = ["accuracy"])
discriminator.summary()


G0 = k.layers.Input(shape = [1,100])
G1 = k.layers.Dense(7*7*256, use_bias = False, input_shape = (100,))(G0)
G2 = k.layers.BatchNormalization()(G1)
G3 = k.layers.LeakyReLU()(G2)
G4 = k.layers.Reshape((7,7,256))(G3)
G5 = k.layers.Conv2DTranspose(128, (5,5), strides = (1,1), padding='same')(G4)
G6 = k.layers.BatchNormalization()(G5)
G7 = k.layers.LeakyReLU()(G6)
G8 = k.layers.Conv2DTranspose(64, (5,5), strides = (2,2), padding='same')(G7)
G9 = k.layers.BatchNormalization()(G8)
G10 = k.layers.LeakyReLU()(G9)
G11 = k.layers.Conv2DTranspose(1, (5,5), strides = (2,2), padding='same', activation='tanh')(G10)


generator = k.models.Model(inputs = [G0], outputs=[G11])
generator.summary()

cross_entropy = k.losses.BinaryCrossentropy(from_logits=True)

def discriminator_loss(real_output, fake_output):
	real_loss = cross_entropy(tf.ones_like(real_output), real_output)
	fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
	total_loss = real_loss + fake_loss
	return total_loss

def generator_loss(fake_output):
	return cross_entropy(tf.ones_like(fake_output), fake_output)

generator_optimizer = k.optimizers.Adam(1e-4)
discriminator_optimizer = k.optimizers.Adam(1e-4)

for i in range(2000):
	print('step ', i)
	with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
		noise = tf.random.normal([1,1,100])		
		generated_image = generator(noise,training=True)
		real_output = discriminator(X[i].reshape(1,28,28), training = True)
		fake_output = discriminator(generated_image, training=True)
		
		gen_loss = generator_loss(fake_output)
		disc_loss = discriminator_loss(real_output, fake_output)

	gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
	gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
	
	generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
	discriminator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))

	

#model.add(k.layers.Conv2D(64,kernel_size=(3,3), strides = (2,2), padding='same', activation ="relu", input_shape=[28,28,1]))
#model.add(k.layers.Conv2D(64, kernel_size = (3,3), strides = (2,2), padding = 'same', activation = 'relu'))
#model.add(k.layers.Flatten())
#model.add(k.layers.Dense(10, activation='softmax'))
#model.summary()
#model.compile(loss="categorical_crossentropy", optimizer='adam', metrics=["accuracy"])
#model.fit(X,Y_, batch_size=128, epochs=1, validation_split=0.1)



#model = k.models.Sequential()
#model.add(k.Input(shape = [28,28,1]))
#model.add(k.layers.Conv2D(64,kernel_size=(3,3), strides = (2,2), padding='same', activation ="relu", input_shape=[28,28,1]))
#model.add(k.layers.Conv2D(64, kernel_size = (3,3), strides = (2,2), padding = 'same', activation = 'relu'))
#model.add(k.layers.Flatten())
#model.add(k.layers.Dense(10, activation='softmax'))
#model.summary()
#model.compile(loss="categorical_crossentropy", optimizer='adam', metrics=["accuracy"])
#model.fit(X,Y_, batch_size=128, epochs=1, validation_split=0.1)

#def input_fn():
	#split = tfds.Split.TRAIN
	#dataset = tfds.load('iris', split = split, as_supervised=True)
	#dataset = dataset.map(lambda features, labels: ({'dense_input':features}, labels))
	#dataset.batch(32).repeat()
	#return dataset

#$for features_batch, labels_batch in input_fn().take(1):
#	print(features_batch)
#	print(labels_batch)
