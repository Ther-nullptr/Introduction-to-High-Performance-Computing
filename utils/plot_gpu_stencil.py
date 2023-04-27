import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    x = list(range(5))
    y_256 = [441.591196, 444.711849, 415.873894, 363.057206, 360.759357]
    y_384 = [469.428652, 470.274611, 469.122651, 468.509651, 406.851275]
    y_512 = [468.452767, 468.810584, 468.049605, 449.508509, 362.065074]

    plt.figure()

    plt.xlabel('block size') 
    my_label = ['(8, 4)', '(8, 8)', '(16, 8)', '(32, 8)', '(32, 32)'] 
    plt.xticks(ticks=[0, 1, 2, 3, 4], labels=my_label, rotation=0, ha='right') 
    plt.ylabel('GFLOPS') 
    plt.plot(x, y_256, marker='x', linestyle='--') 
    plt.plot(x, y_384, marker='x', linestyle='--') 
    plt.plot(x, y_512, marker='x', linestyle='--') 
    plt.legend(['256', '384', '512']) 
    plt.savefig('plot.png') 
