dataset_path = "../pan18-author-profiling-training-2018-02-27"
features_path_faceRecognition = "../feature_extractors/extracted-features/author_images_face_recognition-all.p"
features_path_objectDetection = "../feature_extractors/extracted-features/author_images_yolo-all.p"
features_path_globalFeatures = "../feature_extractors/extracted-features/author_images_global_features-all.p"

def getSplitFeatures(featuresId, splitId, splitPath, languages):
	'''
    Returns a list of the image features of a  selected split An image split is a percentage of data reserved for image classifier 
	training (currently 80% of data are reserved for image classifier training)
	@param featuresId: The id of the features you want to retrieve. Possible values are currently 'face_recognition', 'object_detection', 'global_features'.
	@param splitId: The split to retrieve. Possible values are 70 (70% of the image data), 20 (20% of the image data), 10 (10% of the image data)
	@param splitPath: The path to the split directory
	@param languages: If you want to retrieve a specific language for your split, add it to this array. The Possible values are 'ar', 'en', 'es'. Ex: languages = ['ar', 'en']
    '''
	import pickle
	
	# Loading the desired image features
	features_path = ''
	if featuresId == 'face_recognition':
		features_path = features_path_faceRecognition
	elif featuresId == 'object_detection':
		features_path = features_path_objectDetection
	elif featuresId == 'global_features':
		features_path = features_path_globalFeatures
	
	featuresByAuthor = dict()
	with open(features_path , "rb" ) as input_file:
		featuresByAuthor = pickle.load(input_file)
	
	# Keeping only the features from the desired split
	splitAuthors = getSplitAuthors(splitId, splitPath, languages)
	features_SelectedSplit = dict()
	for author in featuresByAuthor:
		if author in splitAuthors:
			features_SelectedSplit[author] = featuresByAuthor[author]
			
	return features_SelectedSplit
	
def getSplitAuthors(splitId, splitPath, languages):
	'''
    Returns a list of authors in a split An image split is a percentage of data reserved for image classifier 
	training (currently 80% of data are reserved for image classifier training)
	@param splitId: The split to retrieve. Possible values are 70 (70% of the image data), 20 (20% of the image data), 10 (10% of the image data)
	@param splitPath: The path to the split directory
	@param language: If you want to retrieve a specific language for your split, add it to this array. The Possible values are 'ar', 'en', 'es'. Ex: languages = ['ar', 'en']
    '''
	import pickle
	
	# Loading all splits
	ar_split = dict()
	with open(splitPath + '/ar.pkl'  , "rb" ) as input_file:
		ar_split = pickle.load(input_file)
	en_split = dict()
	with open(splitPath + '/en.pkl'  , "rb" ) as input_file:
		en_split = pickle.load(input_file)
	es_split = dict()
	with open(splitPath + '/es.pkl'  , "rb" ) as input_file:
		es_split = pickle.load(input_file)
	
	# Selecting only the desired languages from splits
	languageSelectedSplit = dict()
	for language in languages:
		if language == 'ar':
			languageSelectedSplit.update(ar_split)
		elif language == 'en':
			languageSelectedSplit.update(en_split)
		elif language == 'es':
			languageSelectedSplit.update(es_split)
		else:
			print("Error utilsImageTraining.py (function getSplit): Wrong language value in the languages tab")
	
	# Selecting only the desired part of the split we want to retrieve
	languageAndIdSelectedSplit = []
	for author in languageSelectedSplit:
		if languageSelectedSplit[author] == splitId:
			languageAndIdSelectedSplit.append(author)
			
	return languageAndIdSelectedSplit

def getGenderDict(languages):
	'''
	The goal of this function is to retrieve author's gender from the dataset truthfiles.
    Returns a dict in which each author id is associated to a gender.
	@param languages: The languages you you want the gender ids from.
    '''
	gender_dict = dict()
	for language in languages:	
		truthFilepath_currentLanguage = dataset_path + '/' + language + '/' + language + '.txt'			
		gender_dict.update(parse_gender_dict(truthFilepath_currentLanguage))
		
	return gender_dict

def parse_gender_dict(truthFilePath):
	'''
	Private function used by getGenderDict.
    '''
	import pickle
	
	with open(truthFilePath) as f:
		content = f.readlines()
		content = [x.strip() for x in content]
		
	genders = dict()
	# Female label is 0 ; Male label is 1
	for author_info in content:
		infos = author_info.split(':::')
		current_author_gender = None
		if(infos[1] == 'female'):
			current_author_gender = 0
		else:
			current_author_gender = 1
		genders[infos[0]] = current_author_gender
	
	return genders

def createDataset(featuresByAuthorSelectedSplit, gender_dict):
	'''
	Creating a dict with an incremental id as key and the array [features, author_gender] as value, for each image.
	Private function used by getXandY.
	@featuresByAuthorSelectedSplit: The features selected for the split. It is a dict such as id is authorId and value are the features for each image (another dict).
	'''
	dataset = dict()
	i=0
	for author in featuresByAuthorSelectedSplit:
		author_id = author
		author_gender = gender_dict[author_id]
		for imageIndex in featuresByAuthorSelectedSplit[author_id]:
			features = featuresByAuthorSelectedSplit[author_id][imageIndex]
			dataset[i] = [features, author_gender]
			i += 1
			
	return dataset
	
def getXandY(featuresId, splitId, splitPath, languages):
	'''
	Returns a X and an y for a given set of features.
	@param featuresId: The id of the features you want to retrieve X and y from. Possible values are currently 'face_recognition', 'object_detection', 'global_features'.
	@param splitId: The split to retrieve. Possible values are 70 (70% of the image data), 20 (20% of the image data), 10 (10% of the image data)
	@param splitPath: The path to the split directory
	@param languages: If you want to work on a specific language, add it to this array. The Possible values are 'ar', 'en', 'es'. Ex: languages = ['ar', 'en']
    '''
	# Loading features for the selected split and language
	featuresByAuthorSelectedSplit = getSplitFeatures(featuresId, splitId, splitPath, languages)
	
	# Loading the genders of authors from truth files
	gender_dict = getGenderDict(languages)
	
	# Creating a dict with an incremental id as key and the array [features, author_gender] as value, for each image
	dataset = createDataset(featuresByAuthorSelectedSplit, gender_dict)
	
	# Getting X and Y vectors from the dataset
	import pandas as pd
	import numpy as np
	
	dataset_array = np.asarray(list(dataset.values()))
	images_tweets = pd.DataFrame(dataset_array, columns=['features', 'label',])
	X = np.array(list(images_tweets.features))
	y = np.array(list(images_tweets.label), dtype=">i1")
	
	return X, y

def trainAndSaveLowClassifiers(X, y, classifierType):
	'''
	Trains and save low classifiers, given X and y.
	@param X: Feature vector
	@param y: Prediction vector
	@param classifierType: The type of classifier to train. Possible values are 'color-histogram', 'face-detection', 'lbp', 'object-detection'.
	@param language: If you want to retrieve a specific language for your training, add it to this array. The Possible values are 'ar', 'en', 'es'.
    '''
	from sklearn.model_selection import cross_val_score
	from sklearn.naive_bayes import MultinomialNB
	from sklearn.ensemble import RandomForestClassifier
	from sklearn.tree import DecisionTreeClassifier
	from sklearn.calibration import CalibratedClassifierCV
	from sklearn.neural_network import MLPClassifier
	from sklearn.svm import LinearSVC
	from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
	from sklearn.tree import DecisionTreeClassifier
	
	print('Training classifiers for ' + classifierType + ' -- This may take some time..')
	
	# classifiers = [MultinomialNB(), RandomForestClassifier(random_state=0, n_estimators=10, n_jobs=-1), CalibratedClassifierCV(LinearSVC(random_state=0)), DecisionTreeClassifier(), MLPClassifier(), AdaBoostClassifier(), GradientBoostingClassifier()]
	classifiers = [MultinomialNB(), RandomForestClassifier(random_state=0, n_estimators=10, n_jobs=-1), CalibratedClassifierCV(LinearSVC(random_state=0)), DecisionTreeClassifier()]
	names = ['multiNB', 'rforest', 'linearSVC', 'DecisionTree', 'MLP', 'AdaBoost', 'GradientBoosting']
	
	# Training and saving low classifiers for the current image feature
	import pickle
	lowClfPath = 'trained-classifiers/low-classifiers/'  + classifierType + '/low'
	currentClfIndex = 0
	while currentClfIndex < len(classifiers):
		print('Training', names[currentClfIndex], '...')
		clf = classifiers[currentClfIndex]
		clf.fit(X, y)
		pickle.dump( clf, open( lowClfPath + '/' + classifierType + '-' + names[currentClfIndex] + '-low.p', "wb" ) )
		currentClfIndex += 1
	
	
def trainAndSaveMetaClassifier(X_metaTraining, y_meta, classifierType):
	'''
	Trains a meta classifier from a training feature vector, built from data dedicated to the meta classifier training, called X_metaTraining.
	@param X_metaTraining: Feature vector for the meta classifier training. Predictions from low layers will be done from this vector.
	@param y_meta: Truth vector regarding the X_metaTraining vector
	@param classifierType: The type of classifier to train. Possible values are 'color-histogram', 'face-detection', 'lbp', 'object-detection'.
    '''
	input_meta_image = getInputMetaImageFromLowClassifiers(X_metaTraining, classifierType)
	
	# Training the meta classifier from the outputs of low classifier for the current feature
	from sklearn.model_selection import cross_val_score
	from sklearn.naive_bayes import MultinomialNB
	from sklearn.ensemble import RandomForestClassifier
	from sklearn.tree import DecisionTreeClassifier
	from sklearn.calibration import CalibratedClassifierCV
	from sklearn.svm import LinearSVC
	
	print('Training meta classifier for ' + classifierType + ' -- This may take some time..')
	
	classifiers = [MultinomialNB(), RandomForestClassifier(random_state=0, n_estimators=20, n_jobs=-1, max_features=None), DecisionTreeClassifier(random_state=0, max_features=None), LinearSVC(random_state=0)]
	
	best_classifier = None
	best_accuracy = 0
	best_stdDev = 0
	
	for clf in classifiers:
		current_scores = cross_val_score(clf, input_meta_image, y_meta, cv=20, scoring='accuracy', n_jobs=2, verbose=1)
		if current_scores.mean() > best_accuracy:
			best_accuracy = current_scores.mean()
			best_classifier = clf
			best_stdDev = current_scores.std()
	
	
	# Training the classifier on the whole data
	if str(type(best_classifier)) == "<class 'sklearn.svm.classes.LinearSVC'>":
		best_classifier = CalibratedClassifierCV(best_classifier)
	best_classifier.fit(input_meta_image, y_meta)
	
	# Saving the classifier
	import pickle
	pickle.dump( best_classifier, open( 'trained-classifiers/low-classifiers/'  + classifierType + '/meta/' + classifierType + '-meta.p', "wb" ) )
	
	print('Best classifier saved for meta image:', best_classifier)
	print('Best accuracy:', best_accuracy)
	print('Standard deviation', best_stdDev)
	
	
def getInputMetaImageFromLowClassifiers(X_metaTraining, classifierType):
	'''
	Returns an array containing prediction vectors given by the low layers.
	@param X_metaTraining: Feature vector for the meta classifier training. Predictions from low layers will be done from this vector.
	@param classifierType: The type of classifier to train. Possible values are 'color-histogram', 'face-detection', 'lbp', 'object-detection'.
    '''
	
	lowClfPath = 'trained-classifiers/low-classifiers/'  + classifierType + '/low'

	# Loading trained low classifiers
	from os import listdir
	from os.path import isfile, join
	import pickle
	classifierFileNames = [f for f in listdir(lowClfPath) if isfile(join(lowClfPath, f))]
	
	trainedLowClassifiers = []
	for classifierFileName in classifierFileNames:
		trainedLowClassifiers.append( pickle.load(open( lowClfPath + '/' + classifierFileName, "rb" )) )

	# Getting low classifiers predictions
	allPredictionsFromLowClfs = []
	indexTrainedClf = 0
	for trainedLowClf in trainedLowClassifiers:
		allPredictionsFromLowClfs.append(trainedLowClf.predict_proba(X_metaTraining))
	
	# For all low classifiers predictions, we get the i predicition result and append it in a vector
	input_meta_image = []
	i=0
	while i < len(allPredictionsFromLowClfs[0]):
		current_input_entry = []
		for currentLowClfPredictions in allPredictionsFromLowClfs:
			current_input_entry.append(currentLowClfPredictions[i][0]) # Proba female
			current_input_entry.append(currentLowClfPredictions[i][1]) # Proba male
		input_meta_image.append(current_input_entry)
		i += 1
	
	return input_meta_image