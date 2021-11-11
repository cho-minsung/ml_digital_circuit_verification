# machine learning application to digital circuits verification
import os
import subprocess
import ltspice  # data parsing
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from PyLTSpice.LTSpiceBatch import SimCommander


if __name__ == '__main__':
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

    data = {"nmos length": np.zeros(10**4),
            "nmos width": np.zeros(10**4),
            "pmos length": np.zeros(10**4),
            "pmos width": np.zeros(10**4),
            "switching point": np.zeros(10**4)}

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





