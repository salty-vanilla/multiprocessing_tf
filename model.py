import os
import tensorflow as tf
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.losses import categorical_crossentropy
from tensorflow.python.keras.metrics import categorical_accuracy


class MLP:
    def __init__(self,
                 input_shape=(784, ),
                 nb_classes=10,
                 optimizer=tf.train.GradientDescentOptimizer(learning_rate=1e-4)):
        # create graph
        self.input_ = tf.placeholder(tf.float32, [None] + list(input_shape))
        self.output = self.build(self.input_, nb_classes)
        self.t = tf.placeholder(tf.float32, self.output.get_shape())
        self.loss = tf.reduce_mean(categorical_crossentropy(self.t, self.output))
        self.acc = tf.reduce_mean(categorical_accuracy(self.t, self.output))

        self.optimizer = optimizer.minimize(self.loss)
        self.saver = tf.train.Saver()
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def build(self, x, nb_classes):
        _x = Dense(256, activation='relu')(x)
        _x = Dense(nb_classes, activation='softmax')(_x)
        return _x

    def fit(self, data_generator, nb_epoch, model_dir):
        batch_size = data_generator.batch_size
        nb_sample = data_generator.n

        # calucuate steps per a epoch
        steps_per_epoch = nb_sample // batch_size
        if nb_sample % batch_size != 0:
            steps_per_epoch += 1

        # fit loop
        for epoch in range(1, nb_epoch+1):
            for step in range(steps_per_epoch):
                image_batch, label_batch = data_generator()
                _, loss, acc = self.sess.run([self.optimizer, self.loss, self.acc],
                                             feed_dict={self.input_: image_batch,
                                                        self.t: label_batch})
        self.save(model_dir)

    def save(self, model_dir):
        os.makedirs(model_dir, exist_ok=True)
        self.saver.save(self.sess, os.path.join(model_dir, 'model.ckpt'))
