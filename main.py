# machine learning application to digital circuits verification
import os
import subprocess
import ltspice  # data parsing
import matplotlib.pyplot as plt
import numpy as np
from PyLTSpice.LTSpiceBatch import SimCommander


if __name__ == '__main__':
    schematic_path = "main.asc"
    inverter_path = "INV_20_10.asc"
    with open(inverter_path, 'r') as reader:
        data = reader.readlines()
    newdata = data
    print(newdata[31])

    # parsing NMOS L and W
    temp = newdata[31].split('l=')
    initial_nmos_length, initial_nmos_width = temp[1].split('w=')
    initial_nmos_length = int(initial_nmos_length.split('n')[0])
    if initial_nmos_width.find('n') != -1:
        initial_nmos_width = int(initial_nmos_width.split('n')[0])
    elif initial_nmos_width.find('u') != -1:
        pmos_width = int(initial_nmos_width.split('u')[0]) * 1000
    print(initial_nmos_length, initial_nmos_width)

    # parsing PMOS
    temp = newdata[38].split('l=')
    initial_pmos_length, initial_pmos_width = temp[1].split('w=')
    initial_pmos_length = int(initial_pmos_length.split('n')[0])
    if initial_pmos_width.find('n') != -1:
        initial_pmos_width = int(initial_pmos_width.split('n')[0])
    elif initial_pmos_width.find('u') != -1:
        initial_pmos_width = int(initial_pmos_width.split('u')[0]) * 1000
    print(initial_pmos_length, initial_pmos_width)

    # initial_nmos_length = 50
    # initial_nmos_width = 500
    # initial_pmos_length = 50
    # initial_pmos_width = 1000

    increment = 50  # increment of 50ns

    for h in range(10):  # nmos length sweep
        for i in range(10):  # nmos width sweep
            for j in range(10):  # pmos length sweep
                for k in range(10):  # pmos width sweep
                    temp = newdata[31].split('l=')
                    newdata[31] = temp[0] + "l=" + str(initial_nmos_length + increment * h) + "n w=" + str(
                        initial_nmos_width + increment * i) + "n\n"
                    temp = newdata[38].split('l=')
                    newdata[38] = temp[0] + "l=" + str(initial_pmos_length + increment * j) + "n w=" + str(
                        initial_pmos_width + increment * k) + "n\n"

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
                    print(time[vsp_index])




