file_name = "/ist-nas/users/palakonk/singularity_logs/allpoints_mini.csv"
# file_name = "/ist-nas/users/palakonk/singularity/home/palakons/from_scratch/allpoints_mini_2.csv"
file_name = "/ist-nas/users/palakonk/singularity/home/palakons/from_scratch/allpoints_mini_3.csv"
#read csv file
import pandas as pd
df = pd.read_csv(file_name, sep=",", header=0)

#how many lines
print("Number of lines: ", len(df))

#print the first 5 rows
print(df.head())

#mean and std of x
mean_x = df['x'].mean()
std_x = df['x'].std()
print("Mean of x: ", mean_x)
print("Std of x: ", std_x)

#list all sensors
s = df['sensor'].unique()
print("Sensors: ", s)

# for sensor in s:
#     #mean and std of x from each sensor
#     df_sensor = df[df['sensor'] == sensor]
#     mean_x_sensor = df_sensor['x'].mean()
#     std_x_sensor = df_sensor['x'].std()
#     print("Mean of x from ", sensor, ": ", mean_x_sensor)
#     print("Std of x from ", sensor, ": ", std_x_sensor)
#     #min max
#     min_x_sensor = df_sensor['x'].min()
#     max_x_sensor = df_sensor['x'].max()
#     print("Min of x from ", sensor, ": ", min_x_sensor)
#     print("Max of x from ", sensor, ": ", max_x_sensor)
#     #count
#     count_sensor = df_sensor['x'].count()

def plot_mean_sd(df,fname):
    import matplotlib.pyplot as plt
    import numpy as np
    
    #plot heat map of mean, horizontal axis : sensor, vertical axis : scene

    # make subfix 2 by 3, for mean and std, and for x y and z
    fig, axs = plt.subplots(2, 3, figsize=(16, 9))


    for j, col in enumerate(['x', 'y', 'z']):
        for i, stat in enumerate(['mean', 'std']):
            # mean and std of x y z
            if stat == 'mean':
                data = df.groupby(['sensor', 'scene'])[col].mean().unstack()
            else:
                data = df.groupby(['sensor', 'scene'])[col].std().unstack()
            
            # plot heat map
            im = axs[i, j].imshow(data, cmap='hot', interpolation='nearest')
            axs[i, j].set_title(f'{stat.capitalize()} of {col}')
            axs[i, j].set_ylabel('Sensor')
            axs[i, j].set_xlabel('Scene')
            axs[i, j].set_xticks(np.arange(len(data.columns)))
            axs[i, j].set_yticks(np.arange(len(data.index)))
            axs[i, j].set_xticklabels([a[:5] for a in data.columns], rotation=90)
            axs[i, j].set_yticklabels([a[6:] for a in data.index])
            # add color bar
            cbar = fig.colorbar( im,ax=axs[i, j])

            # print(data)
            # scene              1eb6c0c7adad4cf29e8da13dcf8ef16b  32d2bcf46e734dffb14fe2e0a823d059  454fa18d64d94009914137ce901c8e7c  ...  e43ae3c9b670423d89fe92170f0c87e9  e9f4a50a1b0a41a492edf7a4870f594c  fa2acde00fe3439786ccd631c78fd641
            # sensor                                                                                                                   ...                                                                                                      
            # RADAR_LEFT_BACK                           84.884518                         75.163548                         66.249585  ...                         70.660224                         77.657113                         62.531659
            # RADAR_LEFT_FRONT                          73.557447                         75.323743                         68.927008  ...                         73.545812                         53.454490                         66.711511
            # RADAR_LEFT_SIDE                           67.352904                         69.283275                         42.738750  ...                         79.582176                         58.821635                         58.312681
            # RADAR_RIGHT_BACK                          83.817651                         68.591866                         53.018300  ...                         72.651221                         47.650258                         58.985634
            # RADAR_RIGHT_FRONT                         79.760827                         78.153494                         74.708743  ...                         76.826671                         71.867109                         72.511096
            # RADAR_RIGHT_SIDE                          87.367178                         65.869349                         34.021193  ...                         57.286531                         40.157020                         52.005316
            #calculate min max of data
            min_val = data.min().min()
            max_val = data.max().max()
            # print(f"min: {min_val}, max: {max_val}",col,stat)
            # add value annotation in each cell as well
            for x in range(len(data.columns)):
                for y in range(len(data.index)):
                    axs[i, j].text( x,y, f"{data.iloc[y, x]:.0f}", ha='center', va='center', color='black' if data.iloc[y, x] > (max_val+min_val)/2 else 'white', fontsize=8)

    # adjust layout
    plt.tight_layout()
    # save the figure
    plt.savefig(fname)

    
fname_plot =  file_name.replace(".csv", ".png")
plot_mean_sd(df, fname_plot)