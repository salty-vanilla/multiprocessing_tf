import threading
import time
import tensorflow as tf
from data_generator import DataGenerator
from model import MLP


def operation(coord, nb_epoch=10, model_dir='./model'):
    model = MLP()
    data_gen = DataGenerator()
    model.fit(data_gen, nb_epoch, model_dir)


def main():
    coord = tf.train.Coordinator()
    threads = [threading.Thread(target=operation, args=(coord, 30, 'model_{}'.format(i)))
               for i in range(10)]

    for t in threads:
        time.sleep(1)
        t.start()

    coord.join(threads)
    print('Complete')


if __name__ == '__main__':
    main()