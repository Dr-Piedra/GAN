import numpy as np 
#import tensorflow as tf 
import os
import random
from itertools import product
from keras.models import Model
from keras.layers import Dense, Lambda, Flatten, BatchNormalization, LeakyReLU, Add, Dropout
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, Input
from keras.layers import UpSampling2D, Conv2DTranspose, Reshape, Cropping2D, ZeroPadding2D
from keras import optimizers
from keras import regularizers
from keras.losses import mse
from keras import backend as K
from keras.utils import plot_model
import pandas as pd


#tf.enable_eager_execution()

img_width, img_height, img_channels = 360, 181, 16
regterm =.1
data_directory = '/p/cwfs/drpiedra/newer_data/box/'
sample_number = range(12000)
feature_name = 's11.npy'
batch_number = 1

def sampling(args):
    """Reparameterization trick by sampling from an isotropic unit Gaussian.
    # Arguments
        args (tensor): mean and log of variance of Q(z|X)
    # Returns
        z (tensor): sampled latent vector
    """

    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    # by default, random_normal has mean = 0 and std = 1.0
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon



def conv3x3(input_layer,numel_filters):
	CL_1 = Conv2D(numel_filters, (3,3), padding='same',activation=None,kernel_regularizer=regularizers.l2(regterm),bias_regularizer=regularizers.l2(regterm))(input_layer)
	CL_2 = BatchNormalization(axis=-1,momentum = .5)(CL_1)
	CL_3 = LeakyReLU(alpha=.3)(CL_2)
	CL_4 = Conv2D(numel_filters, (3,3), padding='same',activation=None,kernel_regularizer=regularizers.l2(regterm),bias_regularizer=regularizers.l2(regterm))(CL_3)
	CL_5 = BatchNormalization(axis=-1,momentum = .5)(CL_4)
	CL_6 = Add()([CL_1,CL_5])
	CL_7 = LeakyReLU(alpha=.3)(CL_6)
	return CL_7

def deconv3x3(input_layer,numel_filters):
	CL_1 = Conv2DTranspose(numel_filters, (3,3), padding='same',activation=None,kernel_regularizer=regularizers.l2(regterm),bias_regularizer=regularizers.l2(regterm))(input_layer)
	CL_2 = BatchNormalization(axis=-1,momentum = .5)(CL_1)
	CL_3 = LeakyReLU(alpha=.3)(CL_2)
	CL_4 = Conv2DTranspose(numel_filters, (3,3), padding='same',activation=None,kernel_regularizer=regularizers.l2(regterm),bias_regularizer=regularizers.l2(regterm))(CL_3)
	CL_5 = BatchNormalization(axis=-1,momentum = .5)(CL_4)
	CL_6 = Add()([CL_1,CL_5])
	CL_7 = LeakyReLU(alpha=.3)(CL_6)
	return CL_7
    
def maxpool2x2(input_layer):
	CL_1 = MaxPooling2D(pool_size=(2,2),padding='valid')(input_layer)
	return CL_1

def upsampling2x2(input_layer):
	CL_1 = UpSampling2D(size=(2,2),interpolation='nearest')(input_layer)
	return CL_1

def denselayer(input_layer, units):
	CL_1 = Dense(units,activation='sigmoid')(input_layer)
	CL_2 = LeakyReLU(alpha=.3)(CL_1)
	return CL_2

def flattenlayer(input_layer):
	CL_1 = Flatten()(input_layer)
	return CL_1

def croppinglayer(input_layer, units):
	CL_1 = Cropping2D(cropping = units)(input_layer)
	return CL_1

def paddinglayer(input_layer, units):
	CL_1 = ZeroPadding2D(padding = units)(input_layer)
	return CL_1

def reshapelayer(input_layer, units):
	CL_1 = Reshape(target_shape = units)(input_layer)
	return CL_1

def dropoutlayer(input_layer, percentage):
	CL_1 = Dropout(rate = percentage)(input_layer)
	return CL_1

def gen_sample():
	while True:
		batch_filenames = np.random.choice(sample_number,batch_number)
		batch = []        
		for folder in batch_filenames:			
			sample_directory = data_directory + '/%i/'%folder
			data = []
			for (i,j) in product(range(1,5), range(1,5)):
				if i == 1 and j == 1:
					s_name = 's11_norm.npy'
				else:
					s_name = 's%i%i.npy'%(i,j)
				s_path = sample_directory + s_name	
				if os.path.exists(s_path):
					sij = np.load(s_path)
					data.append(sij)
			batch.append(data)
		try: 
			data_batch = np.array(batch).reshape(batch_number,360,181,img_channels)
		except:
			continue

		yield(data_batch, None)

#generator = tf.data.Dataset.from_generator(gen_sample, output_types=tf.float64, output_shapes=(None,361,181,1))
#iterator = generator.repeat().make_one_shot_iterator()
#features = iterator.get_next()
latent_dim = 2

Image_input = Input(shape = (img_width, img_height, img_channels)) 

#### this is the encoder model
S0 = croppinglayer(Image_input, units = ((0,0),(1,0)))
S1 = conv3x3(S0,28)
S2 = maxpool2x2(S1)
S3 = conv3x3(S2,28)
S4 = maxpool2x2(S3)
S6 = conv3x3(S4,14)
S12 = flattenlayer(S6)
S12_1 = dropoutlayer(S12, 0.25)
S13 = denselayer(S12_1,20)
S13_1 = dropoutlayer(S13, 0.1)
End_mean = Dense(latent_dim, name='End_mean')(S13_1)
End_log_var = Dense(latent_dim, name='End_log_var')(S13)
z = Lambda(sampling, output_shape=(latent_dim,), name='z')([End_mean, End_log_var])


#this is the decoder model
#latent_input =i Input(shape = (latent_dim,), name = 'z_sampling')
#S14 = denselayer(latent_input,20)
S14 = denselayer(z, 20)
S14_1 = dropoutlayer(S14, 0.1)
S15 = denselayer(S14_1,56700)
S15_1 = dropoutlayer(S15, 0.75)
S15_2 = reshapelayer(S15_1, units=(90,45,14))
S16 = upsampling2x2(S15_2)
S17 = deconv3x3(S16,14) 
S18 = upsampling2x2(S17)
#S18 = paddinglayer(S18, units = ((1,0),(2,1))) 
#S18_2 = croppinglayer(S19_1, units = ((0,0),(1,0)))
#S19 = deconv3x3(S18,1)
S19 = Conv2DTranspose(16, (3,3), padding='same',activation='sigmoid',kernel_regularizer=regularizers.l2(regterm),bias_regularizer=regularizers.l2(regterm))(S18)
End = paddinglayer(S19, units = ((0,0),(1,0))) 

#encoder = Model(inputs=[Image_input],outputs=[End_mean, End_log_var, z], name = 'encoder')
#encoder.summary()

#decoder = Model(inputs = [latent_input], outputs= [End], name='decoder')
#decoder.summary()

#decoded_outputs = decoder(encoder(Image_input)[2])
vae = Model(inputs=[Image_input], outputs = [End], name='vae_mlp')
vae.summary()


kl_loss = 1 + End_log_var - K.square(End_mean) - K.exp(End_log_var)
kl_loss = K.sum(kl_loss, axis=-1)
kl_loss *= -0.5
reconstruction_loss = mse(Image_input, End)
#reconstruction_loss *= original_dim
vae_loss = K.mean(reconstruction_loss + kl_loss)
vae.add_loss(vae_loss)
adam = optimizers.Adam(lr = 0.001, beta_1 = 0.9, beta_2 = 0.999)
vae.compile(optimizer=adam)
#plot_model(vae,
               #to_file='vae_mlp.png',
               #show_shapes=True)

<<<<<<< HEAD
history = vae.fit_generator(gen_sample(), epochs = 1, steps_per_epoch = 2)
vae.save('multi_model.h5')
vae.save_weights('multi_weights.h5')
=======
vae.fit_generator(gen_sample(), epochs = 1000, steps_per_epoch = 30)
vae.save('model.h5')
vae.save_weights('weights.h5')
>>>>>>> e46f1601b0fa2664ade7fbee5e33cf18e1098ce7

history_df = pd.DataFrame(history.history)
with open(hist.csv, 'w') as f:
	history_df.to_csv(f)


#a=gen_sample()

#b=next(a)
#vae.fit(b[0])

"""
def architecture(Image_input):
    S1 = conv3x3(Image_input,28)
    S2 = maxpool2x2(S1)
    S3 = conv3x3(S2,28)
    S4 = maxpool2x2(S3)
    S6 = conv3x3(S4,14)
    S7 = maxpool2x2(S6)
    S8 = conv3x3(S7,14)
    S9 = maxpool2x2(S8)
    S10 = conv3x3(S9,7)
    End = maxpool2x2(S10)  
    return End

autoencoder = Model(inputs = [Image_input], outputs=[architecture(Image_input)])

#def encoder(Image_input):
"""
