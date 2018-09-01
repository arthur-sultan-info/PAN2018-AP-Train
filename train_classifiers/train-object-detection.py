import numpy as np
from scipy.sparse import csr_matrix
	
def X_ToSparseMatrix(X_train):
	
	
	columns = []
	rows = []
	values = []
	rowIndex = 0

	convertLabelToInt = dict()
	labelToIntCount = 0

	for observation in X_train:
		for objectLabel in observation['labels']:
			if objectLabel not in convertLabelToInt:
				convertLabelToInt[objectLabel] = labelToIntCount
				labelToIntCount += 1
			columns.append(convertLabelToInt[objectLabel])
			rows.append(rowIndex)
			values.append(observation['labels'][objectLabel])
		
		rowIndex += 1 # next observation item

	row  = np.array(rows)
	col  = np.array(columns)
	data = np.array(values)
	numberOfRows = len(X_train)
	numberOfColumns = len(convertLabelToInt)
	resultSparseMatrix = csr_matrix((data, (row, col)), shape=(numberOfRows, numberOfColumns))

	return (resultSparseMatrix, convertLabelToInt)
	

def X_ToSparseMatrix_objectDetection(X_train, convertLabelToInt):
    columns = []
    rows = []
    values = []
    rowIndex = 0

    labelToIntCount = 0
    
    for observation in X_train:
        for objectLabel in observation['labels']:
            if objectLabel in convertLabelToInt:
                columns.append(convertLabelToInt[objectLabel])
                rows.append(rowIndex)
                values.append(observation['labels'][objectLabel])
        
        rowIndex += 1 # next observation item
    
    row  = np.array(rows)
    col  = np.array(columns)
    data = np.array(values)
    numberOfRows = len(X_train)
    numberOfColumns = len(convertLabelToInt)
    resultSparseMatrix = csr_matrix((data, (row, col)), shape=(numberOfRows, numberOfColumns))
    
    return resultSparseMatrix, convertLabelToInt
	
	
def train(options):
	import utilsImageTraining as imageUtils
	
	# Getting X and y for the desired features and split
	X, y = imageUtils.getXandY(featuresId='object_detection', splitId=70, splitPath='../splits/splitting-lowStacking', languages=options['languages'])
	X, convertLabelToInt = X_ToSparseMatrix(X)
	
	# Training and saving the low classfiers for face recognition
	imageUtils.trainAndSaveLowClassifiers(X, y, 'object-detection')
	
	# Getting X_meta and y_meta to train the meta classifier for face recognition
	X_meta, y_meta = imageUtils.getXandY(featuresId='object_detection', splitId=30, splitPath='../splits/splitting-lowStacking', languages=options['languages'])
	X_meta, convertLabelToInt = X_ToSparseMatrix_objectDetection(X_meta, convertLabelToInt)
	
	# Training and saving the meta classfier for face recognition
	imageUtils.trainAndSaveMetaClassifier(X_meta, y_meta, 'object-detection')
	
	# Saving the classifier
	import pickle
	pickle.dump( convertLabelToInt, open(options['convertLabelToInt_path'], "wb") )
	

	
if __name__ == "__main__":
	options = {
		'convertLabelToInt_path': './labels_object_detection/convertLabelToInt.p',
		'languages': ['ar', 'en', 'es']
	}
	
	train(options)