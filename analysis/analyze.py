import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os.path
import keras

file_path = '../results/' 
save_path = '../figures/'
scores0_file = 'revealcancel_0_scores_mean.csv'
scores1_file = 'revealcancel_1_scores_mean.csv'
#label_file = 'new_test_data_tr_978.csv'
label_file = '../data-05-31-2018/formatted_full_data.csv'
i=1
while os.path.exists(save_path+'analysis_network'+str(i)+'.png'):
    i+=1
network_dir = '../networks/network'+str(i)+'/'
model_name = 'nt3'

print 'Loading data...'
data0 = pd.read_csv(network_dir+scores0_file, sep='\t').values[:,1:]
data1 = pd.read_csv(network_dir+scores1_file, sep='\t').values[:,1:]
data_frame = pd.read_csv(file_path+label_file, sep=',', header=None).values
print 'done'
data = data_frame[:,1:]
label = data_frame[:,0].astype(int)

print 'score for 0 shape:', data0.shape
print 'score for 1 shape:', data1.shape 
print 'sample shape:', label.shape

normal_indices = label == 0
tumor_indices = label == 1

total0 = np.sum(data0, axis=1)
total1 = np.sum(data1, axis=1)

norm_tot0 = total0[normal_indices]
norm_tot1 = total1[normal_indices]

tum_tot0 = total0[tumor_indices]
tum_tot1 = total1[tumor_indices]

accuracy = open('{}/{}_accuracy.txt'.format(network_dir, model_name),'r').read()
plt.title('sum scores for model with ' + accuracy + '% accuracy')
plt.scatter(norm_tot0, norm_tot1, color = 'blue', label='normal (0)')
plt.scatter(tum_tot0, tum_tot1, color = 'red', label = 'tumor (1)')
plt.legend()
plt.savefig(save_path+'analysis_network'+str(i)+'.png')
#plt.show()
