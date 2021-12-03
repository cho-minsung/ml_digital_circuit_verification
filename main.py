# machine learning application to digital circuits verification
import os
import sys
import subprocess
import ltspice  # data parsing
import matplotlib.pyplot as plt
import pandas as pd
pd.set_option('display.max_columns', None)
import numpy as np
np.set_printoptions(threshold=np.inf)
from PyLTSpice.LTSpiceBatch import SimCommander
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def get_data():
    schematic_path = "main.asc"
    inverter_path = "INV_20_10.asc"
    with open(inverter_path, 'r') as reader:
        data = reader.readlines()
    newdata = data

    # parsing NMOS L and W
    temp = newdata[31].split('l=')
    initial_nmos_length, initial_nmos_width = temp[1].split('w=')
    initial_nmos_length = int(initial_nmos_length.split('n')[0])
    if initial_nmos_width.find('n') != -1:
        initial_nmos_width = int(initial_nmos_width.split('n')[0])
    elif initial_nmos_width.find('u') != -1:
        pmos_width = int(initial_nmos_width.split('u')[0]) * 1000

    # parsing PMOS
    temp = newdata[38].split('l=')
    initial_pmos_length, initial_pmos_width = temp[1].split('w=')
    initial_pmos_length = int(initial_pmos_length.split('n')[0])
    if initial_pmos_width.find('n') != -1:
        initial_pmos_width = int(initial_pmos_width.split('n')[0])
    elif initial_pmos_width.find('u') != -1:
        initial_pmos_width = int(initial_pmos_width.split('u')[0]) * 1000

    initial_nmos_length = 50
    initial_nmos_width = 500
    initial_pmos_length = 50
    initial_pmos_width = 1000

    increment = 50  # increment of 50ns

    data = {"nmos length": np.zeros(10 ** 4),
            "nmos width": np.zeros(10 ** 4),
            "pmos length": np.zeros(10 ** 4),
            "pmos width": np.zeros(10 ** 4),
            "switching point": np.zeros(10 ** 4)}

    count = 0
    for h in range(10):  # nmos length sweep
        for i in range(10):  # nmos width sweep
            for j in range(10):  # pmos length sweep
                for k in range(10):  # pmos width sweep
                    nmos_len = initial_nmos_length + increment * h
                    nmos_width = initial_nmos_width + increment * i
                    pmos_len = initial_pmos_length + increment * j
                    pmos_width = initial_pmos_width + increment * k

                    temp = newdata[31].split('l=')
                    newdata[31] = temp[0] + "l=" + str(nmos_len) + "n w=" + str(
                        nmos_width) + "n\n"
                    temp = newdata[38].split('l=')
                    newdata[38] = temp[0] + "l=" + str(pmos_len) + "n w=" + str(
                        pmos_width) + "n\n"

                    with open(inverter_path, 'w') as writer:
                        writer.writelines(newdata)

                    ltspice_command = "C:\Program Files\LTC\LTspiceXVII\XVIIx64.exe -Run -b " + schematic_path
                    subprocess.run(ltspice_command)

                    filepath = 'main.raw'
                    lt = ltspice.Ltspice(filepath)
                    lt.parse()

                    time = lt.get_time()
                    input_voltage = lt.get_data('V(in)')
                    output_voltage = lt.get_data('V(out)')
                    temp = min(output_voltage, key=lambda x: abs(x - 0.5))
                    vsp_index = np.where(output_voltage == temp)
                    vsp = time[vsp_index]

                    data['nmos length'][count] = nmos_len
                    data['nmos width'][count] = nmos_width
                    data['pmos length'][count] = pmos_len
                    data['pmos width'][count] = pmos_width
                    data['switching point'][count] = vsp
                    print(count)
                    count += 1

    df = pd.DataFrame(data, columns=['nmos length', 'nmos width', 'pmos length', 'pmos width', 'switching point'])
    df.to_csv('data.csv')


if __name__ == '__main__':
    column_name = ['nmos length', 'nmos width', 'pmos length', 'pmos width', 'switching point']
    raw_dataset = pd.read_csv('data.csv', names=column_name, skipinitialspace=True)
    dataset = raw_dataset.copy()

    train_dataset = dataset.sample(frac=0.8, random_state=0)
    test_dataset = dataset.drop(train_dataset.index)

    # sns.pairplot(train_dataset[['nmos length', 'nmos width', 'pmos length', 'pmos width']], diag_kind='kde')
    # plt.show()

    print(train_dataset.describe().transpose())

    train_features = train_dataset.copy()
    test_features = test_dataset.copy()

    train_labels = train_features.pop('switching point')
    test_labels = test_features.pop('switching point')

    train_dataset.describe().transpose()

    normalizer = tf.keras.layers.Normalization(axis=-1)
    normalizer.adapt(np.array(train_features))
    print(normalizer.mean.numpy())
    first = np.array(train_features[:1])

    with np.printoptions(precision=2, suppress=True):
        print('First example:', first)
        print('Normalized:', normalizer(first).numpy())

    nmos_length = np.array(train_features['nmos length'])
    nmos_length_normalizer = layers.Normalization(input_shape=[1, ], axis=None)
    nmos_length_normalizer.adapt(nmos_length)

    nmos_length_model = tf.keras.Sequential([
        nmos_length_normalizer,
        layers.Dense(units=1)
    ])

    nmos_length_model.summary()

    print(nmos_length_model.predict(nmos_length[:10]))

    nmos_length_model.compile(
        optimizer=tf.optimizers.Adam(learning_rate=0.1),
        loss='mean_absolute_error'
    )

    history = nmos_length_model.fit(
        train_features['nmos length'],
        train_labels,
        epochs=100,
        # Suppress logging.
        verbose=0,
        # Calculate validation results on 20% of the training data.
        validation_split=0.2)

    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch
    print(hist.tail())

    def plot_loss(history):
        plt.plot(history.history['loss'], label='loss')
        plt.plot(history.history['val_loss'], label='val_loss')
        plt.ylim([0, 0.02])
        plt.xlabel('Epoch')
        plt.ylabel('Error [switching point]')
        plt.legend()
        plt.grid(True)
        plt.show()

    plot_loss(history)

    test_results = {}
    test_results['nmos_length'] = nmos_length_model.evaluate(
        test_features['nmos length'],
        test_labels, verbose=0
    )

    x = tf.linspace(0.0, 250, 251)
    y = nmos_length_model.predict(x)


    def plot_nmos_length(x, y):
        plt.scatter(train_features['nmos length'], train_labels, label='Data')
        plt.plot(x, y, color='k', label='Predictions')
        plt.xlabel('nmos length')
        plt.ylabel('switching point')
        # plt.ylim(top=max(y), bottom=0)
        plt.legend()
        plt.show()

    plot_nmos_length(x, y)

    linear_model = tf.keras.Sequential([
        normalizer,
        layers.Dense(units=1)
    ])

    linear_model.predict(train_features[:10])

    print(linear_model.layers[1].kernel)

    linear_model.compile(
        optimizer=tf.optimizers.Adam(learning_rate=0.1),
        loss='mean_absolute_error')

    history = linear_model.fit(
        train_features,
        train_labels,
        epochs=100,
        # Suppress logging.
        verbose=0,
        # Calculate validation results on 20% of the training data.
        validation_split=0.2)

    plot_loss(history)


    def build_and_compile_model(norm):
        model = keras.Sequential([
            norm,
            layers.Dense(64, activation='relu'),
            layers.Dense(64, activation='relu'),
            layers.Dense(1)
        ])
        model.compile(loss='mean_absolute_error',
                      optimizer=tf.keras.optimizers.Adam(0.001))
        return model

    dnn_model = build_and_compile_model(normalizer)
    print(dnn_model.summary())

    history = dnn_model.fit(
        train_features,
        train_labels,
        validation_split=0.2,
        verbose=0, epochs=100)

    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch
    print(hist.tail())

    plot_loss(history)
