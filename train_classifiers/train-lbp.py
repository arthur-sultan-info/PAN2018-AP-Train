
def X_ToSparseMatrix(X_train):
	import numpy as np
	from scipy.sparse import csr_matrix

	columns = []
	rows = []
	values = []
	rowIndex = 0

	color_histogram_flattened_length = len(X_train[0]['local_binary_patterns'])

	for observation in X_train:
		observation_color_histogram = observation['local_binary_patterns']
		color_histogram_index = 0
		while (color_histogram_index < len(observation_color_histogram)):
			columns.append(color_histogram_index)
			rows.append(rowIndex)
			values.append(observation_color_histogram[color_histogram_index])
			color_histogram_index += 1
		
		rowIndex += 1 # next observation item

	row  = np.array(rows)
	col  = np.array(columns)
	data = np.array(values)
	numberOfRows = len(X_train)
	numberOfColumns = color_histogram_flattened_length
	resultSparseMatrix = csr_matrix((data, (row, col)), shape=(numberOfRows, numberOfColumns))

	return resultSparseMatrix

	
def train(options):
	import utilsImageTraining as imageUtils
	
	# Getting X and y for the desired features and split
	X, y = imageUtils.getXandY(featuresId='global_features', splitId=70, splitPath='../splits/splitting-lowStacking', languages=options['languages'])
	X = X_ToSparseMatrix(X)
	
	# Training and saving the low classfiers for face recognition
	imageUtils.trainAndSaveLowClassifiers(X, y, 'lbp')
	
	# Getting X_meta and y_meta to train the meta classifier for face recognition
	X_meta, y_meta = imageUtils.getXandY(featuresId='global_features', splitId=30, splitPath='../splits/splitting-lowStacking', languages=options['languages'])
	X_meta = X_ToSparseMatrix(X_meta)
	
	# Training and saving the meta classfier for face recognition
	imageUtils.trainAndSaveMetaClassifier(X_meta, y_meta, 'lbp')
	

if __name__ == "__main__":
	options = {
		'languages': ['ar', 'en', 'es']
	}
	
	train(options)