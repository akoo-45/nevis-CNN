@article{He2015,
  author = {Kaiming He and Xiangyu Zhang and Shaoqing Ren and Jian Sun},
  title = {Deep Residual Learning for Image Recognition},
  journal = {arXiv preprint arXiv:1512.03385},
  year = {2015}
}


from __future__ import division
from larcv import larcv
from larcv.dataloader2 import larcv_threadio
import numpy as np
import os,sys,time

# tensorflow/gpu start-up configuration
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES']='2'
import tensorflow as tf

TUTORIAL_DIR     = '.'
TRAIN_IO_CONFIG  = os.path.join(TUTORIAL_DIR, 'tf/io_train.cfg')
TEST_IO_CONFIG   = os.path.join(TUTORIAL_DIR, 'tf/io_test.cfg' )
TRAIN_BATCH_SIZE = 10
TEST_BATCH_SIZE  = 100
LOGDIR           = 'resnet_log'
ITERATIONS       = 5000
SAVE_SUMMARY     = 20
SAVE_WEIGHTS     = 100

# Check log directory is empty
train_logdir = os.path.join(LOGDIR,'train')
test_logdir  = os.path.join(LOGDIR,'test')
if not os.path.isdir(train_logdir): os.makedirs(train_logdir)
if not os.path.isdir(test_logdir):  os.makedirs(test_logdir)
if len(os.listdir(train_logdir)) or len(os.listdir(test_logdir)):
  sys.stderr.write('Error: train or test log dir not empty...\n')
  raise OSError

#step 0: IO
#
# for "train" data set
train_io = larcv_threadio()  # create io interface
train_io_cfg = {'filler_name' : 'TrainIO',
                'verbosity'   : 10,
                'filler_cfg'  : TRAIN_IO_CONFIG}
train_io.configure(train_io_cfg)   # configure
train_io.start_manager(TRAIN_BATCH_SIZE) # start read thread
time.sleep(2)
train_io.next()

# for "test" data set
test_io = larcv_threadio()   # create io interface
test_io_cfg = {'filler_name' : 'TestIO',
               'verbosity'   : 10,
               'filler_cfg'  : TEST_IO_CONFIG}
test_io.configure(test_io_cfg)   # configure
test_io.start_manager(TEST_BATCH_SIZE) # start read thread
time.sleep(2)
test_io.next()

# 
# Step 1: Define Network
#
import tensorflow.contrib.slim as slim # TODO
import tensorflow.python.platform # TODO

import six
from keras.models import Model
from keras.layers import (
    Input,
    Activation,
    Dense,
    Flatten
)
from keras.layers.convolutional import (
    Conv2D,
    MaxPooling2D,
    AveragePooling2D
)
from keras.layers.merge import add
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras import backend as K


def _bn_relu(input):
    """Helper to build a BN -> relu block
    """
    norm = BatchNormalization(axis=CHANNEL_AXIS)(input)
    return Activation("relu")(norm)


def _conv_bn_relu(**conv_params):
    """Helper to build a conv -> BN -> relu block
    """
    filters = conv_params["filters"]
    kernel_size = conv_params["kernel_size"]
    strides = conv_params.setdefault("strides", (1, 1)) # Overrides strides?
    kernel_initializer = conv_params.setdefault("kernel_initializer", "he_normal")
    padding = conv_params.setdefault("padding", "same") # Padding, SAME?????????
    kernel_regularizer = conv_params.setdefault("kernel_regularizer", l2(1.e-4))

    print("Entering _conv_bn_relu")
    #input_shape = K.int_shape(input)
    #print(input_shape)
    def f(input):

        conv = Conv2D(filters=filters, kernel_size=kernel_size,
                      strides=strides, padding=padding,
                      kernel_initializer=kernel_initializer,
                      kernel_regularizer=kernel_regularizer)(input)
        return _bn_relu(conv)

    return f

# TODO: reduce code later by duplicating with _conv_bn_relu
def _conv_bn(**conv_params):
    """Helper to build a conv -> BN -> relu block
    """
    filters = conv_params["filters"]
    kernel_size = conv_params["kernel_size"]
    strides = conv_params.setdefault("strides", (1, 1)) # Overrides strides?
    kernel_initializer = conv_params.setdefault("kernel_initializer", "he_normal")
    padding = conv_params.setdefault("padding", "same") # Padding, SAME?????????
    kernel_regularizer = conv_params.setdefault("kernel_regularizer", l2(1.e-4))

    def f(input):
        conv = Conv2D(filters=filters, kernel_size=kernel_size,
                      strides=strides, padding=padding,
                      kernel_initializer=kernel_initializer,
                      kernel_regularizer=kernel_regularizer)(input)
        return BatchNormalization(axis=CHANNEL_AXIS)(conv)

    return f


def _bn_relu_conv(**conv_params):
    """Helper to build a BN -> relu -> conv block.
    This is an improved scheme proposed in http://arxiv.org/pdf/1603.05027v2.pdf
    """
    filters = conv_params["filters"]
    kernel_size = conv_params["kernel_size"]
    strides = conv_params.setdefault("strides", (1, 1))
    kernel_initializer = conv_params.setdefault("kernel_initializer", "he_normal")
    padding = conv_params.setdefault("padding", "same")
    kernel_regularizer = conv_params.setdefault("kernel_regularizer", l2(1.e-4))

    def f(input):
        activation = _bn_relu(input)
        return Conv2D(filters=filters, kernel_size=kernel_size,
                      strides=strides, padding=padding,
                      kernel_initializer=kernel_initializer,
                      kernel_regularizer=kernel_regularizer)(activation)

    return f

# probably nothing to change 
def _shortcut(input, residual):
    """Adds a shortcut between input and residual block and merges them with "sum"
    """
    # Expand channels of shortcut to match residual.
    # Stride appropriately to match residual (width, height)
    # Should be int if network architecture is correctly configured.
    print("Entering shortcut")
    input_shape = K.int_shape(input)
    residual_shape = K.int_shape(residual)
    print (input_shape)
    print (residual_shape) # TODO remove later 
    print ("Input channels ", input_shape[CHANNEL_AXIS])
    print ("Residual channels ", residual_shape[CHANNEL_AXIS])
    stride_width = int(round(input_shape[ROW_AXIS] / residual_shape[ROW_AXIS]))
    stride_height = int(round(input_shape[COL_AXIS] / residual_shape[COL_AXIS]))


    equal_channels = input_shape[CHANNEL_AXIS] == residual_shape[CHANNEL_AXIS]

    shortcut = input
    # 1 X 1 conv if shape is different. Else identity.
    if stride_width > 1 or stride_height > 1 or not equal_channels:
        print("The number of filters ", residual_shape[CHANNEL_AXIS]) # TODO
        shortcut = Conv2D(filters=residual_shape[CHANNEL_AXIS],
                          kernel_size=(1, 1),
                          strides=(stride_width, stride_height),
                          padding="valid",
                          kernel_initializer="he_normal",
                          kernel_regularizer=l2(0.0001))(input)

    return add([shortcut, residual])

# 64, 1,1, false 
def _residual_block(block_function, filters, repetitions, is_first_layer=False):
    """Builds a residual block with repeating bottleneck blocks.
    """
    def f(input): # ??
        for i in range(repetitions): # [0] = range 1
            init_strides = (1, 1) # really confusing?! 
            if i == 0 and not is_first_layer:  # NOTE: this was confusing to interpret
                init_strides = (2, 2) # Why strides = (2,2) for the rest of the layers?
                 # WHERE IS THIS IN THE MODEL (possible optimization) **
            input = block_function(filters=filters, init_strides=init_strides, # WHY? 
                                   is_first_block_of_first_layer=(is_first_layer and i == 0))(input)
        return input

    return f

def basic_block(filters, init_strides=(1, 1), is_first_block_of_first_layer=False):
    """Basic 3 X 3 convolution blocks for use on resnets with layers <= 34.
    Follows improved proposed scheme in http://arxiv.org/pdf/1603.05027v2.pdf
    """
    def f(input):

        if is_first_block_of_first_layer:




           # don't repeat bn->relu since we just did bn->relu->maxpool
            conv1 = Conv2D(filters=filters, kernel_size=(3, 3),
                           strides=init_strides,
                           padding="same",
                           kernel_initializer="he_normal",
                           kernel_regularizer=l2(1e-4))(input)
        else:
            conv1 = _bn_relu_conv(filters=filters, kernel_size=(3, 3),
                                  strides=init_strides)(input)

        residual = _bn_relu_conv(filters=filters, kernel_size=(3, 3))(conv1)
        return _shortcut(input, residual)

    return f


def bottleneck(filters, init_strides=(1, 1), is_first_block_of_first_layer=False):
    """Bottleneck architecture for > 34 layer resnet.
    Follows improved proposed scheme in http://arxiv.org/pdf/1603.05027v2.pdf
    Returns:
        A final conv layer of filters * 4
    """
    def f(input):

        # if is_first_block_of_first_layer:
            # don't repeat bn->relu since we just did bn->relu->maxpool # you still did it... (possible optimization) TODO **
        #     conv_1_1 = Conv2D(filters=filters, kernel_size=(1, 1),
        #                       strides=init_strides,
        #                       padding="same",
        #                       kernel_initializer="he_normal",
        #                       kernel_regularizer=l2(1e-4))(input)
        # else:
            # revert later ** changed all 3  _bn_relu_conv --> _conv_bn_relu (possible optimization) TODO
            # see if padding needs to change TODO padding = same for ALL 
        print("Entering bottleneck stage")
        conv_1_1 = _conv_bn_relu(filters=filters, kernel_size=(1, 1),
                                     strides=init_strides)(input) # Pass on strides!?



        conv_3_3 = _conv_bn_relu(filters=filters, kernel_size=(3, 3))(conv_1_1)
        # filters = filters * 4 is true if you check the ethereon model for Resnet 50 
        # might have to edit ** last layer doesn't have relu...  ** TODO
        # residual = _conv_bn_relu(filters=filters * 4, kernel_size=(1, 1))(conv_3_3) # THIS HAS DIFFERENT PADDING THOUGH ?? Does padding = same do the trick??
        residual = _conv_bn(filters=filters * 4, kernel_size=(1, 1))(conv_3_3)
        print("Filter used for the residual layer ",  filters * 4)
        return _shortcut(input, residual)

    return f

# image_dim_ordering => image_data_format ** 
def _handle_dim_ordering():
    global ROW_AXIS
    global COL_AXIS
    global CHANNEL_AXIS
    if K.image_data_format() == 'tf':
        ROW_AXIS = 1
        COL_AXIS = 2
        CHANNEL_AXIS = 3 # Changed this from 1, 2, 3 because the tuple takes on none, 64, 64, 256 apparently DOUBLE CHECK WHY
    else:
        print("The K backend engine says not tf")
        print("Then what is it?", K.image_data_format())
        CHANNEL_AXIS = 3
        ROW_AXIS = 1
        COL_AXIS = 2


def _get_block(identifier):
    if isinstance(identifier, six.string_types):
        res = globals().get(identifier)
        if not res:
            raise ValueError('Invalid {}'.format(identifier))
        return res
    return identifier


class ResnetBuilder(object):
    @staticmethod
    def build(input_shape, num_outputs, block_fn, repetitions):
        """Builds a custom ResNet like architecture.
        Args:
            input_shape: The input shape in the form (nb_channels, nb_rows, nb_cols)
            num_outputs: The number of outputs at final softmax layer
            block_fn: The block function to use. This is either `basic_block` or `bottleneck`.
                The original paper used basic_block for layers < 50
            repetitions: Number of repetitions of various block units.
                At each block unit, the number of filters are doubled and the input size is halved
        Returns:
            The keras `Model`.
        """
        _handle_dim_ordering()
        if len(input_shape) != 3:
            raise Exception("Input shape should be a tuple (nb_channels, nb_rows, nb_cols)")

        # Permute dimension order if necessary *
        # if K.image_data_format() == 'tf':
        #     input_shape = (input_shape[1], input_shape[2], input_shape[0]) # rows, cols, samples? *?

        # Load function from str if needed.
        block_fn = _get_block(block_fn)

        input = Input(shape=input_shape) # creation of keras tensor  
        conv1 = _conv_bn_relu(filters=64, kernel_size=(7, 7), strides=(2, 2))(input) # padding = 3? *? TODO
        pool1 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="same")(conv1)

        block = pool1
        filters = 64
        for i, r in enumerate(repetitions): # handy-dandy!
            block = _residual_block(block_fn, filters=filters, repetitions=r, is_first_layer=(i == 0))(block) #?
            filters *= 2

        # Last activation
        # I think this is just _relu... for the last block  but will leave it TODO
        block = _bn_relu(block)

        # Classifier block
        # changed from strides=(1,1) to (18,18) ** TODO don't know why kernel size doesn't show up in Kazu's... 
        block_shape = K.int_shape(block)
        pool2 = AveragePooling2D(pool_size=(block_shape[ROW_AXIS], block_shape[COL_AXIS]),
                                 strides=(18, 18))(block) # where is the kernel?? 7?? TODO
        print("Entering flattening")
        flatten1 = Flatten()(pool2) # this isn't in the model TODO maybe this is the inner product 
        print("Entering softmax")
        dense = Dense(units=num_outputs, kernel_initializer="he_normal",
                      activation="softmax")(flatten1)

        model = Model(inputs=input, outputs=dense)
        print("returning model")
        # WHERE IS ACCURACY ** TODO
        return model

    # @staticmethod
    # def build_resnet_18(input_shape, num_outputs):
    #     return ResnetBuilder.build(input_shape, num_outputs, basic_block, [2, 2, 2, 2])

    # **
    @staticmethod
    def build_resnet_14b(input_shape, num_outputs):
        return ResnetBuilder.build(input_shape, num_outputs, bottleneck, [1, 1, 1, 1])

    # @staticmethod
    # def build_resnet_34(input_shape, num_outputs):
    #     return ResnetBuilder.build(input_shape, num_outputs, basic_block, [3, 4, 6, 3])

    # @staticmethod
    # def build_resnet_50(input_shape, num_outputs):
    #     return ResnetBuilder.build(input_shape, num_outputs, bottleneck, [3, 4, 6, 3])

    # @staticmethod
    # def build_resnet_101(input_shape, num_outputs):
    #     return ResnetBuilder.build(input_shape, num_outputs, bottleneck, [3, 4, 23, 3])

    # @staticmethod
    # def build_resnet_152(input_shape, num_outputs):

# Deleted 

#
# Step 2: Build network + define loss & solver
#
# retrieve dimensions of data for network construction
dim_data  = train_io.fetch_data('train_image').dim() # [10, 256, 256, 1]
dim_label = train_io.fetch_data('train_label').dim()
print('dim_data', dim_data)
print('dim_label', dim_label)
# define place holders
data_tensor    = tf.placeholder(tf.float32, [None, dim_data[1] * dim_data[2] * dim_data[3]], name='image')
label_tensor   = tf.placeholder(tf.float32, [None, dim_label[1]], name='label')
data_tensor_2d = tf.reshape(data_tensor, [-1,dim_data[1],dim_data[2],dim_data[3]],name='image_reshape')
print('data_tensor_2d', data_tensor_2d.shape)

# Let's keep 10 random set of images in the log
tf.summary.image('input',data_tensor_2d,10)
# build net -- changed this TODO
# net = build(inputs=data_tensor_2d, trainable=True, num_class=dim_label[1], debug=False)

net = ResnetBuilder.build_resnet_14b(input_shape=(256, 256, 1), num_outputs=5)
# net is keras.engine.training.Model 
# Define accuracy

net.compile(optimizer='rmsprop', 
            loss='categorical_crossentropy', 
            metrics=['accuracy'])

# TODO 
model.fit(X_train, Y_train,
          batch_size=batch_size,
          nb_epoch=nb_epoch,
          validation_data=(X_test, Y_test),
          shuffle=True,
          callbacks=[lr_reducer, early_stopper, csv_logger])

x: Numpy array of training data
y: Numpy array of target (label) data
epochs: Integer
verbose = 2


 tf.name_scope('accuracy'):
  correct_prediction = tf.equal(tf.argmax(net,1), tf.argmax(label_tensor,1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  tf.summary.scalar('accuracy', accuracy)
  # Define loss + backprop as training step
  with tf.name_scope('train'):
    print('label_tensor', label_tensor)
    print('logits', net)
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=label_tensor, logits=net))
    tf.summary.scalar('cross_entropy',cross_entropy)
    train_step = tf.train.RMSPropOptimizer(0.00005).minimize(cross_entropy)


#                                                                                                                                      
# Step 3: weight saver & summary writer                                                                                                
#                                                                                                                                      
# Create a bandle of summary                                                                                                           
merged_summary=tf.summary.merge_all()
# Create a session                                                                                                                     
sess = tf.InteractiveSession()
# Initialize variables                                                                                                                 
sess.run(tf.global_variables_initializer())
# Create a summary writer handle                                                                                                       
writer_train=tf.summary.FileWriter(train_logdir)
writer_train.add_graph(sess.graph)
writer_test=tf.summary.FileWriter(test_logdir)
writer_test.add_graph(sess.graph)
# Create weights saver                                                                                                                 
saver = tf.train.Saver()

#
# Step 4: Run training loop
#
for i in range(ITERATIONS):
    train_data  = train_io.fetch_data('train_image').data()
    train_label = train_io.fetch_data('train_label').data()

    feed_dict = { data_tensor  : train_data,
                  label_tensor : train_label }
    # Why doesn't any of this part go through net? ***
    loss, acc, _ = sess.run([cross_entropy, accuracy, train_step], feed_dict=feed_dict)

    if (i+1)%SAVE_SUMMARY == 0:
      # Save train log
      sys.stdout.write('Training in progress @ step %d loss %g accuracy %g          \n' % (i,loss,acc))
      sys.stdout.flush()
      s = sess.run(merged_summary, feed_dict=feed_dict)
      writer_train.add_summary(s,i)

      # Calculate & save test log
      test_data  = test_io.fetch_data('test_image').data()
      test_label = test_io.fetch_data('test_label').data()
      feed_dict  = { data_tensor  : test_data,
                       label_tensor : test_label }
      loss, acc = sess.run([cross_entropy, accuracy], feed_dict=feed_dict)
      sys.stdout.write('Testing in progress @ step %d loss %g accuracy %g          \n' % (i,loss,acc))
      sys.stdout.flush()
      s = sess.run(merged_summary, feed_dict=feed_dict)
      writer_test.add_summary(s,i)

    test_io.next()

    train_io.next()

    if (i+1)%SAVE_WEIGHTS == 0:
      ssf_path = saver.save(sess,'weights/toynet',global_step=i)
      print('saved @',ssf_path)

# inform log directory
print()
print('Run `tensorboard --logdir=%s` in terminal to see the results.' % LOGDIR)
train_io.reset()
test_io.reset()