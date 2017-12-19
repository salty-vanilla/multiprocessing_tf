import threading
import time
import tensorflow as tf
from data_generator import DataGenerator
from model import MLP


NB_PROCESS = 10


def operation(coord, nb_epoch=10, model_dir='./model'):
    model = MLP()
    data_gen = DataGenerator()
    model.fit(data_gen, nb_epoch, model_dir)


def main():
    # create coordinator for managing multi-thread process
    coord = tf.train.Coordinator()
    # create multi-thread process
    threads = [threading.Thread(target=operation, args=(coord, 30, 'model_{}'.format(i)))
               for i in range(NB_PROCESS)]

    # process run
    for t in threads:
        # sleep for process from conflicting
        time.sleep(1)
        t.start()

    coord.join(threads)
    print('Complete')


if __name__ == '__main__':
    main()
