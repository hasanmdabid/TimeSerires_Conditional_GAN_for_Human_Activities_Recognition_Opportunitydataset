""""
This Functon will plot the Epoch vs Losses (Discriminator and generator)
1st we will take the average of d_loss_real , d_loss_fake, and g_loss
2nd we will plot them in the graphs
"""
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

data = pd.read_csv(r'C:\Users\abid1\PycharmProjects\TimeGAN_Tensorflow_2\Conditional_GAN\python\results\CGAN_500_epochs_32Batch_Minority_withoutBatchNormalization_labelSmoothing_conv2d_lstm.csv', sep=';')

data = data.drop(['gen_activation', ' batch_per_epoch'], axis=1)
#data = data.drop(['g_loss'], axis=1)
print(data.head(5))
df = data.groupby(by=[" epochs"]).mean()
print(df.head())
#df.iloc[0:20].plot()
df.plot()
plt.xlabel('Epochs', fontsize=20) # x-axis label with fontsize 15
plt.ylabel('Loss', fontsize=20) # y-axis label with fontsize 15
plt.legend(fontsize="20") 
plt.show()
plt.savefig('HARCGAN_performance_1kEpoch_CONV2D_LSTM.jpg', dpi=500)


