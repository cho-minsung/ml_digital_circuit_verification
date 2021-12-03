# machine learning application to digital circuits verification
import os
import sys
import subprocess
import ltspice  # data parsing
import random
import matplotlib.pyplot as plt
import pandas as pd
pd.set_option('display.max_columns', None)
import numpy as np
from sklearn.model_selection import train_test_split
np.set_printoptions(threshold=np.inf)
from PyLTSpice.LTSpiceBatch import SimCommander
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras import layers, Input, Model


def generate_nums(start, end):
    return random.uniform(start, end)


def get_data():
    database = {
        "NMOS width": np.zeros(1000),
        "PMOS width": np.zeros(1000),
        "Temperature": np.zeros(1000),
        "VDD": np.zeros(1000),
        "Switching point voltage high": np.zeros(1000),
        "Switching point voltage low": np.zeros(1000),
    }

    for a in range(1000):
        # randomly generated a VDD between 0.9 and 1.1.
        vdd = generate_nums(0.9, 1.1)

        # randomly generated 10 NMOS widths between 50 and 500 with an increment of 50.
        nmos_width_list = np.arange(50, 550, 50)
        nmos_width = random.choice(nmos_width_list)

        # randomly generated 10 PMOS widths between 50 and 500 with an increment of 50.
        pmos_width_list = np.arange(50, 1050, 50)
        pmos_width = random.choice(pmos_width_list)

        temperature = generate_nums(-60, 140)

        schmitt_path = "schmitty.asc"
        with open(schmitt_path, 'r') as reader:
            data = reader.readlines()
        newdata = data

        # setting NMOS width
        temp = newdata[93].split('w=')
        newdata[93] = temp[0] + 'w=' + str(nmos_width) + 'n\n'

        # setting PMOS width
        temp = newdata[79].split('w=')
        newdata[79] = temp[0] + 'w=' + str(pmos_width) + 'n\n'

        with open(schmitt_path, 'w') as writer:
            writer.writelines(newdata)

        schematic_path = "main2.asc"
        with open(schematic_path, 'r') as reader:
            data = reader.readlines()
        newdata = data

        # setting VDD
        newdata[26] = "SYMATTR Value " + str(vdd) + '\n'

        # setting temperature
        print(newdata[48])
        newdata[48] = "TEXT -672 32 Left 2 !.temp " + str(temperature) + '\n'

        with open(schematic_path, 'w') as writer:
            writer.writelines(newdata)

        ltspice_command = "C:\Program Files\LTC\LTspiceXVII\XVIIx64.exe -Run -b " + schematic_path
        subprocess.run(ltspice_command)

        filepath = 'main2.raw'
        lt = ltspice.Ltspice(filepath)
        lt.parse()

        time = lt.get_time()
        print(len(time))
        vsp_high = []
        vsp_low = []
        input_voltage = lt.get_data('V(in)')
        output_voltage = lt.get_data('V(out)')
        vsp_index = np.isclose(input_voltage, output_voltage, atol=0.01)
        vsp_list = input_voltage[np.where(vsp_index)]
        for i in range(len(vsp_list)):
            if vsp_list[i] > 0.5:
                vsp_high.append(vsp_list[i])
            elif vsp_list[i] < 0.5:
                vsp_low.append(vsp_list[i])
        print("%d out of 1000" % a)
        print(vsp_high)
        print(vsp_low)
        if not vsp_high or not vsp_low:
            a -= 1
            continue
        else:
            database['NMOS width'][a] = nmos_width
            database['PMOS width'][a] = pmos_width
            database['Temperature'][a] = temperature
            database['VDD'][a] = pmos_width
            database['Switching point voltage high'][a] = vsp_high[0]
            database['Switching point voltage low'][a] = vsp_low[0]

    df = pd.DataFrame(database, columns=[
        'NMOS width',
        'PMOS width',
        'Temperature',
        'VDD',
        'Switching point voltage high',
        'Switching point voltage low'
    ])
    df.to_csv('data2.csv')


if __name__ == '__main__':
    data = np.genfromtxt('data2.csv', delimiter=',')[1:, 1:]
    column_name = ['NMOS width', 'PMOS width', 'Temperature', 'VDD', 'Switching point voltage high',
                   'Switching point voltage low']
    df = pd.DataFrame(data, columns=column_name)

    # plotting
    # print(df.dtypes)
    # sns.heatmap(df.corr())
    # sns.pairplot(df)
    # plt.show()

    train_dataset, test_dataset = train_test_split(df, test_size=0.2, random_state=1)
    train_dataset, val_dataset = train_test_split(train_dataset, test_size=0.2, random_state=1)

    train_stats = train_dataset.describe()
    train_stats.pop('Switching point voltage high')
    train_stats.pop('Switching point voltage low')
    train_stats = train_stats.transpose()

    def norm(x):
        return (x - train_stats['mean']) / train_stats['std']

    def format_output(data):
        y1 = data.pop('Switching point voltage high')
        y1 = np.array(y1)
        y2 = data.pop('Switching point voltage low')
        y2 = np.array(y2)
        return y1, y2

    train_Y = format_output(train_dataset)
    test_Y = format_output(test_dataset)
    val_Y = format_output(val_dataset)
    # print(train_stats)

    norm_train_X = np.array(norm(train_dataset))
    norm_test_X = np.array(norm(test_dataset))
    norm_val_X = np.array(norm(val_dataset))

    inputs = Input(shape=(4,))
    x = Dense(16, activation='relu')(inputs)
    x = Dense(16, activation='relu')(x)
    x = Dense(16, activation='relu')(x)
    x = Dense(16, activation='relu')(x)
    output1 = Dense(1, name='SP_high_output')(x)
    output2 = Dense(1, name='SP_low_output')(x)
    model = Model(
        inputs=inputs,
        outputs=[output1, output2]
    )
    model.compile(
        loss={
            'SP_high_output': 'mse',
            'SP_low_output': 'mse'
        },
        optimizer=tf.keras.optimizers.Adam(0.001),
        metrics={'SP_high_output': tf.keras.metrics.RootMeanSquaredError(),
                 'SP_low_output': tf.keras.metrics.RootMeanSquaredError()}
    )

    print(model.summary())

    history = model.fit(
        norm_train_X,
        train_Y,
        epochs=100,
        batch_size=10,
        validation_data=[norm_test_X, test_Y]
    )

    loss, SP_high_loss, SP_low_loss, SP_high_err, SP_low_err = model.evaluate(x=norm_val_X, y=val_Y)

    print()
    print(f'loss: {loss}')
    print(f'SP_high_loss: {SP_high_loss}')
    print(f'SP_low_loss: {SP_low_loss}')
    print(f'SP_high_err: {SP_high_err}')
    print(f'SP_low_err: {SP_low_err}')

    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch
    print(hist.tail())


    def plot_diff(y_true, y_pred, title=''):
        plt.scatter(y_true, y_pred)
        plt.title(title)
        plt.xlabel('True Values')
        plt.ylabel('Predictions')
        plt.axis('equal')
        plt.axis('square')
        plt.xlim(plt.xlim())
        plt.ylim(plt.ylim())
        plt.plot([-100, 100], [-100, 100])
        plt.show()


    def plot_metrics(metric_name, title):
        plt.title(title)
        plt.plot(history.history[metric_name], color='blue', label=metric_name)
        plt.plot(history.history['val_' + metric_name], color='green', label='val_' + metric_name)
        plt.show()

    Y_pred = model.predict(norm_test_X)
    sp_high_pred = Y_pred[0]
    sp_low_pred = Y_pred[1]
    print(norm_test_X[0:2])
    print("high prediction", sp_high_pred)
    print("low prediction", sp_low_pred)

    # plot_diff(test_Y[0], Y_pred[0], title='High switching point output voltage')
    # plot_diff(test_Y[1], Y_pred[1], title='Low switching point output voltage')
    #
    # plot_metrics(metric_name='SP_high_output_root_mean_squared_error', title='High switching point output voltage RMSE')
    # plot_metrics(metric_name='SP_low_output_root_mean_squared_error', title='Low switching point output voltage RMSE')
    #
    # plot_metrics(metric_name='SP_high_output_loss', title='High switching point output voltage LOSS')
    # plot_metrics(metric_name='SP_low_output_loss', title='Low switching point output voltage LOSS')

    # model.save('./schmitt_model/', save_format='tf')

    # loaded_model = tf.keras.models.load_model('./schmitt_model/')
