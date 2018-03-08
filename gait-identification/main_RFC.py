__author__ = 'jason'

from data_tool import load_training_validation_data
from model.models import RandomForestClassification
from feature.hog import flatten

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

train_view = "090"
val_view = "090"
train_dir = ["nm-01"] 
val_dir = ["nm-02"] 

if __name__ == '__main__':
	train_x, train_y, val_x, val_y = load_training_validation_data(
		train_view=train_view,
		train_dir=train_dir,
		val_view=val_view,
		val_dir=val_dir)	
	train_feature_x = [flatten(x) for x in train_x]
	val_feature_x = [flatten(x) for x in val_x]
	
	model = RandomForestClassification()
	model.fit(x_train=train_feature_x, y_train=train_y)
	predict_y = model.predict(val_feature_x)
	
	print "predict_y: "
	print predict_y

	correct_count = sum(predict_y==val_y)
	accuracy = correct_count*1.0/len(val_y)

	print "train view: %s, val view: %s, accuracy: %d/%d=%.3f" % \
		(train_view, val_view, correct_count, len(val_y), accuracy)
