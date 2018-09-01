from os.path import join
from sklearn.model_selection import train_test_split
from random import uniform
import pickle
import os

def parse_gender_dict(truthFilePath):
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

def splitting(inputPath, outputPath):
	options = {
		'dataset_path': "pan18-author-profiling-training-2018-02-27"
	}
	
	# Parsing gender
	gender_dict = parse_gender_dict(options['dataset_path'] + '/ar/ar.txt')
	gender_dict.update(parse_gender_dict(options['dataset_path'] + '/en/en.txt'))
	gender_dict.update(parse_gender_dict(options['dataset_path'] + '/es/es.txt'))
	
	# Splitting each language
	splitDic = dict()
	splitDic70 = dict()
	split70 = dict()
	split30 = dict()
	splitDicOr = dict()
	for lang in ['ar', 'en', 'es']:
		with open(inputPath + '/' + lang + '.pkl', 'rb') as input_file:
			splitDic = pickle.load(input_file)
		
		splitDicOr[lang] = splitDic
		splitDic70[lang] = dict()
		for author in splitDic:
			if splitDic[author] == 70:
				splitDic70[lang][author] = splitDic[author]
		
		cptDispatchMale = 1
		cptDispatchFemale = 1
		split70[lang] = dict()
		split30[lang] = dict()
		for author in splitDic70[lang]:
			currentAuthorGender = gender_dict[author]
			if currentAuthorGender == 0:					
				if cptDispatchFemale < 4:
					split30[lang][author] = currentAuthorGender
					cptDispatchFemale += 1
				elif cptDispatchFemale < 10:
					split70[lang][author] = currentAuthorGender
					cptDispatchFemale += 1
				else:
					split70[lang][author] = currentAuthorGender
					cptDispatchFemale = 1
				
			else:
				if cptDispatchMale < 4:
					split30[lang][author] = currentAuthorGender
					cptDispatchMale += 1
				elif cptDispatchMale < 10:
					split70[lang][author] = currentAuthorGender
					cptDispatchMale += 1
				else:
					split70[lang][author] = currentAuthorGender
					cptDispatchMale = 1
					
	# Joining splits in one big dict
	split70_30 = dict()
	for lang in split70:
		split70_30[lang] = dict()
		for author in split70[lang]:
			split70_30[lang][author] = 70
		for author in split30[lang]:
			split70_30[lang][author] = 30
	
	# Getting each lang in one dic for saving in one file for each language
	split70_30_ar = split70_30['ar']
	split70_30_en = split70_30['en']
	split70_30_es = split70_30['es']
	
	for lang in ['ar', 'en', 'es']:
		print('----------------')
		print('Language:', lang)
		nbMale = 0
		nbFemale = 0
		for author in split70[lang]:
			if gender_dict[author] == 0:
				nbFemale += 1
			else:
				nbMale += 1
		print('NbMale split 70:', nbMale)
		print('nbFemale split 70:', nbFemale)
		
		nbMale = 0
		nbFemale = 0
		for author in split30[lang]:
			if gender_dict[author] == 0:
				nbFemale += 1
			else:
				nbMale += 1
		print('NbMale split 30:', nbMale)
		print('nbFemale split 30:', nbFemale)
	
	# Saving splits
	pickle.dump( split70_30_ar, open( outputPath + '/ar.pkl', "wb" ) )
	pickle.dump( split70_30_en, open( outputPath + '/en.pkl', "wb" ) )
	pickle.dump( split70_30_es, open( outputPath + '/es.pkl', "wb" ) )
	

if __name__ == "__main__":
    splitting("splits/splitting-low-meta-combine","splits/splitting-lowStacking")