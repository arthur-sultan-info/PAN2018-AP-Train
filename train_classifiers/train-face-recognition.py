
def X_ToSparseMatrix(X_train):
	import numpy as np
	from scipy.sparse import csr_matrix
	
	columns = []
	rows = []
	values = []
	rowIndex = 0

	labelToIntCount = 0

	for observation in X_train:
		if observation['Female'] > 0:
			columns.append(0)
			rows.append(rowIndex)
			values.append(observation['Female'])
		if observation['Male'] > 0:
			columns.append(1)
			rows.append(rowIndex)
			values.append(observation['Male'])
		
		rowIndex += 1 # next observation item

	row  = np.array(rows)
	col  = np.array(columns)
	data = np.array(values)
	numberOfRows = len(X_train)
	numberOfColumns = 2
	resultSparseMatrix = csr_matrix((data, (row, col)), shape=(numberOfRows, numberOfColumns))

	return resultSparseMatrix
	
	
def train(options):
	import utilsImageTraining as imageUtils
	
	# Getting X and y for the desired features and split
	X, y = imageUtils.getXandY(featuresId='face_recognition', splitId=70, splitPath='../splits/splitting-lowStacking', languages=options['languages'])
	X = X_ToSparseMatrix(X)
	
	# Training and saving the low classfiers for face recognition
	imageUtils.trainAndSaveLowClassifiers(X, y, 'face-detection')
	
	# Getting X_meta and y_meta to train the meta classifier for face recognition
	X_meta, y_meta = imageUtils.getXandY(featuresId='face_recognition', splitId=30, splitPath='../splits/splitting-lowStacking', languages=options['languages'])
	X_meta = X_ToSparseMatrix(X_meta)
	
	# Training and saving the meta classfier for face recognition
	imageUtils.trainAndSaveMetaClassifier(X_meta, y_meta, 'face-detection')
	

if __name__ == "__main__":
	options = {
		'languages': ['ar', 'en', 'es']
	}
	
	train(options)