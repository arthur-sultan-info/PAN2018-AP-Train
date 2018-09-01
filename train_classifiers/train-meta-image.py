import numpy as np
import pandas as pd
import pickle

from scipy.sparse import csr_matrix

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

def X_ToSparseMatrix_colorHistogram(X_train):	
    columns = []
    rows = []
    values = []
    rowIndex = 0

    convertLabelToInt = dict()
    color_histogram_flattened_length = len(X_train[0]['color_histogram'])

    for observation in X_train:
        observation_color_histogram = observation['color_histogram']
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
    numberOfColumns = color_histogram_flattened_length + len(convertLabelToInt)
    resultSparseMatrix = csr_matrix((data, (row, col)), shape=(numberOfRows, numberOfColumns))
    
    return (resultSparseMatrix, convertLabelToInt)

def X_ToSparseMatrix_lbp(X_train):	
    columns = []
    rows = []
    values = []
    rowIndex = 0

    convertLabelToInt = dict()
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
    numberOfColumns = color_histogram_flattened_length + len(convertLabelToInt)
    resultSparseMatrix = csr_matrix((data, (row, col)), shape=(numberOfRows, numberOfColumns))
    
    return (resultSparseMatrix, convertLabelToInt)
	
def X_ToSparseMatrix_faceRecognition(X_train):	
	columns = []
	rows = []
	values = []
	rowIndex = 0

	convertLabelToInt = dict()
	labelToIntCount = 0 

	for observation in X_train:
		if 'Female' in observation:
			if observation['Female'] > 0:
				columns.append(0)
				rows.append(rowIndex)
				values.append(observation['Female'])
		else:
			print('No Female')
		if 'Male' in observation:
			if observation['Male'] > 0:
				columns.append(1)
				rows.append(rowIndex)
				values.append(observation['Male'])
		else:
			print('No Male')
		rowIndex += 1 # next observation item

	row  = np.array(rows)
	col  = np.array(columns)
	data = np.array(values)
	numberOfRows = len(X_train)
	numberOfColumns = 2
	resultSparseMatrix = csr_matrix((data, (row, col)), shape=(numberOfRows, numberOfColumns))

	return (resultSparseMatrix, convertLabelToInt)
	
def createDataset(author_images, gender_dict):
	dataset = dict()
	for author in author_images:
		author_id = author
		author_gender = gender_dict[author_id]
		for imageIndex in author_images[author_id]:
			features = author_images[author_id][imageIndex]
			
			dataset[str(author_id + '.' + str(imageIndex))] = [features, author_gender]
	
	return dataset

def getXandYandIds(dataset):
	X = []
	y = []
	ids = []
	for author in dataset:
		X.append(dataset[author][0])
		y.append(dataset[author][1])
		ids.append(author)
	return X, y, ids
	
def train(options):
	import utilsImageTraining as imageUtils
	
	# Loading features for the selected split and language
	author_images_object_detection = imageUtils.getSplitFeatures(featuresId='object_detection', splitId=20, splitPath='../splits/splitting-low-meta-combine', languages=options['languages'])
	author_images_face_recognition = imageUtils.getSplitFeatures(featuresId='face_recognition', splitId=20, splitPath='../splits/splitting-low-meta-combine', languages=options['languages'])
	author_images_global_features = imageUtils.getSplitFeatures(featuresId='global_features', splitId=20, splitPath='../splits/splitting-low-meta-combine', languages=options['languages'])
	
	
	# Concatening all features in a big author_images
	for author in author_images_object_detection:
		if author in author_images_global_features:
			for image_index in author_images_object_detection[author]:
				if image_index in author_images_global_features[author]:
					author_images_object_detection[author][image_index].update(author_images_global_features[author][image_index])
		if author in author_images_face_recognition:
			for image_index in author_images_object_detection[author]:
				if image_index in author_images_face_recognition[author]:
					author_images_object_detection[author][image_index].update(author_images_face_recognition[author][image_index])
	author_images = author_images_object_detection
	
	# Loading the truth file, for each language
	gender_dict = imageUtils.getGenderDict(options['languages'])
	
	# Creating a dict with an incremental id as key and the array [features, author_gender] as value, for each image
	dataset = imageUtils.createDataset(author_images, gender_dict)
	
	# Getting X and Y vectors from the dataset
	X, y, ids = getXandYandIds(dataset)
	
	
	# Converting X to sparse matrix, for each type of feature
	convertLabelToInt = None
	with open(options['convertLabelToInt_path'] , "rb" ) as input_file:
		convertLabelToInt = pickle.load(input_file)
	
	X_objectDetection, convertLabelToInt = X_ToSparseMatrix_objectDetection(X, convertLabelToInt)
	X_colorHistogram, convertLabelToInt = X_ToSparseMatrix_colorHistogram(X)
	X_lbp, convertLabelToInt = X_ToSparseMatrix_lbp(X)
	X_faceRecognition, convertLabelToInt = X_ToSparseMatrix_faceRecognition(X)
	
	# Loading classifiers --
	objectDetectionClf = None
	with open(options['clf_path_object_detection'] , "rb" ) as input_file:
		objectDetectionClf = pickle.load(input_file)
	
	faceRecognitionClf = None
	with open(options['clf_path_face_recognition'] , "rb" ) as input_file:
		faceRecognitionClf = pickle.load(input_file)
	
	lbpClf = None
	with open(options['clf_path_lbp'] , "rb" ) as input_file:
		lbpClf = pickle.load(input_file)
	
	colorHistogramClf = None
	with open(options['clf_path_color_histogram'] , "rb" ) as input_file:
		colorHistogramClf = pickle.load(input_file)
	
	# Making the predictions for each classifier	
	pred_face_recognition = faceRecognitionClf.predict_proba(imageUtils.getInputMetaImageFromLowClassifiers(X_faceRecognition, 'face-detection'))
	pred_color_histogram = colorHistogramClf.predict_proba(imageUtils.getInputMetaImageFromLowClassifiers(X_colorHistogram, 'color-histogram'))
	pred_lbp = lbpClf.predict_proba(imageUtils.getInputMetaImageFromLowClassifiers(X_lbp, 'lbp'))
	pred_object_detection = objectDetectionClf.predict_proba(imageUtils.getInputMetaImageFromLowClassifiers(X_objectDetection, 'object-detection'))
	
	# Reconstructing face recognition (for debug)
	datasetPred = dict()
	i=0
	while i < len(pred_face_recognition):
		datasetPred[ids[i]] = {
			'face_recognition': pred_face_recognition[i],
			'color_histogram': pred_color_histogram[i],
			'lbp': pred_lbp[i],
			'object_detection': pred_object_detection[i]
		}
		i+=1
	
	# Building the meta image input data from the output of the classifiers for each entry
	input_meta_image = []
	i=0
	while i < len(pred_face_recognition):
		current_input_entry = [pred_face_recognition[i][0], pred_face_recognition[i][1],
								pred_color_histogram[i][0], pred_color_histogram[i][1],
								pred_lbp[i][0], pred_lbp[i][1],
								pred_object_detection[i][0], pred_object_detection[i][1]]
		input_meta_image.append(current_input_entry)
		i+=1
	
	# Selecting the best classifier from a 20-fold cross validation
	from sklearn.model_selection import cross_val_score
	from sklearn.naive_bayes import MultinomialNB
	from sklearn.ensemble import RandomForestClassifier
	from sklearn.tree import DecisionTreeClassifier
	from sklearn.calibration import CalibratedClassifierCV
	from sklearn.svm import LinearSVC
	
	print('Training meta image classifier -- This may take some time..')
	
	classifiers = [MultinomialNB(), RandomForestClassifier(random_state=0, n_estimators=20, n_jobs=-1, max_features=None), DecisionTreeClassifier(random_state=0, max_features=None), LinearSVC(random_state=0)]
	
	best_classifier = None
	best_accuracy = 0
	best_stdDev = 0
	
	for clf in classifiers:
		current_scores = cross_val_score(clf, input_meta_image, y, cv=20, scoring='accuracy', n_jobs=2, verbose=1)
		if current_scores.mean() > best_accuracy:
			best_accuracy = current_scores.mean()
			best_classifier = clf
			best_stdDev = current_scores.std()
	
	
	# Training the classifier on the whole data
	if str(type(best_classifier)) == "<class 'sklearn.svm.classes.LinearSVC'>":
		best_classifier = CalibratedClassifierCV(best_classifier)
	best_classifier.fit(input_meta_image, y)
	
	# Saving the classifier
	pickle.dump( best_classifier, open( "trained-classifiers/meta-image-classifier.p", "wb" ) )
	
	print('Best classifier saved for meta image:', best_classifier)
	print('Best accuracy:', best_accuracy)
	print('Standard deviation', best_stdDev)



if __name__ == "__main__":
	options = {
		'features_path_yolo': "../feature_extractors/extracted-features/author_images_yolo-all.p",
		'features_path_global_features': "../feature_extractors/extracted-features/author_images_global_features-all.p",
		'features_path_face_recognition': "../feature_extractors/extracted-features/author_images_face_recognition-all.p",
		'clf_path_object_detection': "./trained-classifiers/low-classifiers/object-detection/meta/object-detection-meta.p",
		'clf_path_lbp': "./trained-classifiers/low-classifiers/lbp/meta/lbp-meta.p",
		'clf_path_color_histogram': "./trained-classifiers/low-classifiers/color-histogram/meta/color-histogram-meta.p",
		'clf_path_face_recognition': "./trained-classifiers/low-classifiers/face-detection/meta/face-detection-meta.p",
		'convertLabelToInt_path': './labels_object_detection/convertLabelToInt.p',
		'dataset_path': "../Min dataset",
		'split_path': "../output/splitting-low-meta-combine",
		'languages': ['ar', 'en', 'es']
	}
	
	train(options)