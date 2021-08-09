import csv
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

def data_plot(folder, filename, label:str, x_label:str, y_label:str):
    # x = []
    # with open(file, 'r') as f:
    #     lines = csv.reader(f, delimiter=',')
    #     yield next(lines)
    #     for row in lines:
    #         x.append(row[0])
    #         # y.append(int(row[1]))
    
    chunk = pd.read_csv(folder+filename, chunksize=1000000)
    pd_df = pd.concat(chunk)
    pd_df = pd_df[210000: 230000]

    pd_df.plot(kind='line', title=label, x=0)
    # plt.plot(x, color = 'g', linestyle = 'solid',label = label)
    
    plt.xticks(rotation = 25)
    # plt.xlabel(x_label)
    # plt.ylabel(y_label)
    plt.title(label, fontsize = 20)
    plt.grid()
    plt.legend()
    plt.show()
    # plt.savefig('output.png')

data_plot("./results/08081851_8/", "step_reward_v13.csv", "Step rewards", "step", "reward")


