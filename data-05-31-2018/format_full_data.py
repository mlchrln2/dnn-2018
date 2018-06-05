import numpy as np
import pandas as pd
import os.path
def main():
	output_file = 'formatted_full_data.csv'
	if os.path.exists(output_file):
		return
	full_data_file = 'full_data_with_labs.csv'
	feature_samples = 'lincs1000.tsv'
	#save_path = '../Benchmarks/Pilot1/NT3/'
	save_path = ''

	print('Loading data...')
	samples = pd.read_csv(feature_samples, sep='\t').values[:,0]
	data_frame = pd.read_csv(full_data_file, sep=',')
	full_samples = data_frame.columns.values[1:]
	print('done')

	subset = np.isin(full_samples, samples)

	data = data_frame.values
	features = data[:,1:]

	labels = data[:,0]
	data = features[:,subset]
	names = full_samples[subset]

	data_frame = np.hstack((labels.reshape((labels.shape[0],1)), data))
	print 'data_frame shape:', data_frame.shape
	output_data = pd.DataFrame(data_frame)

	print('Writing data...')
	output_data.to_csv(save_path+output_file, header=None, index=False)
	print('done')

main()