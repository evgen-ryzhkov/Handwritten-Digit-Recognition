import settings
from scripts.data import DigitsData

from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import GridSearchCV
from sklearn.externals import joblib
import numpy as np
import pandas as pd
import time
from scipy.misc import imread


class DigitClassifier:

	def __init__(self):
		self.X = None
		self.y = None

	def send_train_set(self, train_images, train_labels):
		self.X = train_images
		self.y = train_labels

	def compare_models(self):
		self._run_sgd_classifier()              # the worst accuracy
		self._run_random_forest_classifier()    # the best accuracy / speed ratio
		self._run_kneighbors_classifier()       # the best accuracy, but speed is very slow (~ 1 hour on my laptop)

	def configure_hyperparameters(self):
		print('Grid search was started...')
		clf = RandomForestClassifier()
		param_grid = [
			{'n_estimators': [10, 30, 50, 100, 200], 'criterion': ['gini', 'entropy'], 'min_samples_split': [2, 5, 10]}
		]

		grid_search = GridSearchCV(clf, param_grid, cv=5, scoring='neg_mean_squared_error')
		grid_search.fit(self.X, self.y)
		print('The best hyper parameters = ', grid_search.best_params_)

	def train_final_model(self):
		print('Building the model...')
		# hyperparameters are choosed by grid search
		final_model = RandomForestClassifier(n_estimators=200, criterion='gini', min_samples_split=2)
		final_model.fit(self.X, self.y)
		y_train_pred = cross_val_predict(final_model, self.X, self.y, cv=3)
		self.calculate_model_metrics(self.y, y_train_pred, model_name='Final model (Random forest classifier)')
		return final_model

	@staticmethod
	def save_final_model(final_model):
		try:
			joblib.dump(final_model, settings.FINAL_MODEL_FILE_NAME)
		except IOError:
			raise ValueError('Something wrong with file save operation.')

	@staticmethod
	def get_final_model():
		try:
			return joblib.load(settings.FINAL_MODEL_FILE_NAME)
		except FileNotFoundError:
			raise ValueError('Model file not found!')

	def _run_sgd_classifier(self):
		# SGD model need 2 dimensions data
		X_train_prepared = self.X.reshape(self.X.shape[0], 784)
		sgd_clf = SGDClassifier(random_state=42)
		sgd_clf.fit(X_train_prepared, self.y)
		y_train_pred = cross_val_predict(sgd_clf, X_train_prepared, self.y, cv=3)
		self.calculate_model_metrics(self.y, y_train_pred, model_name='SGD classifier')

	def _run_random_forest_classifier(self):
		start_time = time.time()
		print('Building the model...')
		forest_clf = RandomForestClassifier()
		forest_clf.fit(self.X, self.y)
		y_train_pred = cross_val_predict(forest_clf, self.X, self.y, cv=3)
		end_time = time.time()
		print('Total time =', round(end_time - start_time), 's')
		self.calculate_model_metrics(self.y, y_train_pred, model_name='Random forest classifier')

	def _run_kneighbors_classifier(self):
		# example of counting of time spending for the model work
		start_time = time.time()
		print('Building the model...')
		knb_clf = KNeighborsClassifier()
		knb_clf.fit(self.X, self.y)
		y_train_pred = cross_val_predict(knb_clf, self.X, self.y, cv=3)
		end_time = time.time()
		print('Total time =', end_time - start_time) # 3600sec
		print('Calculating metrics...')
		self.calculate_model_metrics(self.y, y_train_pred, model_name='K neighbors classifier')

	def calculate_model_metrics(self, Y, y_pred, model_name):
		print('Calculating metrics...')
		labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
		precision, recall, fscore, support = precision_recall_fscore_support(
			Y, y_pred,
			labels=labels)

		precision = np.reshape(precision, (10, 1))
		recall = np.reshape(recall, (10, 1))
		fscore = np.reshape(fscore, (10, 1))
		data = np.concatenate((precision, recall, fscore), axis=1)
		df = pd.DataFrame(data)
		df.columns = ['Precision', 'Recall', 'Fscore']
		print(model_name, '\n')
		print(df)

		print('\n Average values')
		print('Precision = ', df['Precision'].mean())
		print('Recall = ', df['Recall'].mean())
		print('F1 score = ', df['Fscore'].mean())

	def classify_by_image(self, image_file_28x28):
		# input images must be 28x28px size
		prepared_image_array = self._prepare_image(image_file_28x28)
		final_model = self.get_final_model()
		predicted_digit = final_model.predict(prepared_image_array.reshape(1, -1))
		return predicted_digit

	@staticmethod
	def _prepare_image(image_file_28x28):
		img_as_arr = imread(image_file_28x28, mode='L')
		normalized_image = (255 - img_as_arr) / 255 # normalize with invert background color
		prepared_image = normalized_image.reshape(-1, 784)[0]
		return prepared_image