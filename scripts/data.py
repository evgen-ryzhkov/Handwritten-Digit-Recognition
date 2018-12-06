import settings
import os
import glob
import mnist
import matplotlib.pyplot as plt
import numpy as np
from scipy.misc import imsave, imread
from scipy.ndimage import gaussian_filter, rotate, zoom
import random


class DigitsData:

	# MNIST data set was separated on train (first 60 0000) and test (last 10 000) sets
	# mnist lib has special methods for getting train and test sets
	def get_train_set(self):
		train_images = mnist.train_images()
		train_labels = mnist.train_labels()
		mixed_indexes = np.random.permutation(60000)
		train_images, train_labels = train_images[mixed_indexes], train_labels[mixed_indexes]
		train_images_prepared = self._prepare_data(train_images)
		return train_images_prepared, train_labels

	def get_test_set(self):
		test_images = mnist.test_images()
		test_labels = mnist.test_labels()
		test_images_prepared = self._prepare_data(test_images)
		return test_images_prepared, test_labels

	@staticmethod
	def show_image_from_data_set(digit_from_data_set):
		try:
			plt.imshow(digit_from_data_set, cmap='Greys')
			plt.show()
		except:
			digit_from_data_set = digit_from_data_set.reshape(28, 28)
			plt.imshow(digit_from_data_set, cmap='Greys')
			plt.show()

	@staticmethod
	def show_image_from_file(file_name):
		try:
			f = imread(file_name)
			plt.imshow(f)
			plt.show()
		except FileNotFoundError:
			raise ValueError('File ' + file_name+ ' not found')



	# saving original images for further creating synthetic data with images operation
	# this code don't use this approach (array operations are used)
	# it's only for training / an example
	@staticmethod
	def save_original_images(images_arr):
		print('Preparing directory...')

		# first need to clean the dir
		old_files = glob.glob(settings.ORIGINAL_IMAGES_DIR + '*.png')
		if old_files:
			for f in old_files:
				os.remove(f)

		# save images
		print('Images saving was started...')
		total_images = str(images_arr.shape[0])
		for idx, img in enumerate(images_arr, start=1):
			img_i = np.array(img).reshape(28, 28)
			file_name = settings.ORIGINAL_IMAGES_FILE_NAME + str(idx) + '.png'
			out = os.path.join(settings.ORIGINAL_IMAGES_DIR, file_name)
			try:
				imsave(out, img_i)
				print('\r', str(idx), ' images are saved from ' + total_images, end='', flush=True)
			except IOError:
				raise ValueError('Something wrong with file save operation.')
		print('\nImages saving finished.')

	def get_extended_data(self, original_data_set, original_labels):

		# due to limited laptop resources synthesize limited data
		#
		# 8 and 9 numbers have the lowest accuracy
		# so we extend data for this numbers
		# and for 5 numbers also a little bit

		# In this example only rotation operation are used because it gave the best results with a little spending of time
		rotated_8_data_set, new_labels_for_rotated_8 = self._get_synthesis_data(original_data_set, original_labels,
		                                                        extending_digit = 8,
		                                                        extending_size = 5000,
                                                                operation_type = 'rotate')
		rotated_8_data_set_2, new_labels_for_rotated_8_2 = self._get_synthesis_data(original_data_set, original_labels,
		                                                        extending_digit=8,
		                                                        extending_size=5000,
		                                                        operation_type='rotate')
		rotated_8_data_set_3, new_labels_for_rotated_8_3 = self._get_synthesis_data(original_data_set, original_labels,
		                                                        extending_digit=8,
		                                                        extending_size=5000,
		                                                        operation_type='rotate')
		rotated_9_data_set, new_labels_for_rotated_9 = self._get_synthesis_data(original_data_set, original_labels,
		                                                        extending_digit=9,
		                                                        extending_size=5000,
		                                                        operation_type='rotate')
		rotated_9_data_set_2, new_labels_for_rotated_9_2 = self._get_synthesis_data(original_data_set, original_labels,
		                                                        extending_digit=9,
		                                                        extending_size=5000,
		                                                        operation_type='rotate')
		rotated_9_data_set_3, new_labels_for_rotated_9_3 = self._get_synthesis_data(original_data_set, original_labels,
		                                                        extending_digit=9,
		                                                        extending_size=5000,
		                                                        operation_type='rotate')
		rotated_5_data_set, new_labels_for_rotated_5 = self._get_synthesis_data(original_data_set, original_labels,
		                                                        extending_digit=5,
		                                                        extending_size=5000,
		                                                        operation_type='rotate')
		rotated_5_data_set_2, new_labels_for_rotated_5_2 = self._get_synthesis_data(original_data_set, original_labels,
		                                                        extending_digit=5,
		                                                        extending_size=5000,
		                                                        operation_type='rotate')

		new_train_set = np.concatenate(
			(original_data_set, rotated_8_data_set, rotated_8_data_set_2, rotated_8_data_set_3,
			 rotated_9_data_set, rotated_9_data_set_2, rotated_9_data_set_3,
			 rotated_5_data_set, rotated_5_data_set_2), axis=0)
		new_labels = np.concatenate(
			(original_labels, new_labels_for_rotated_8, new_labels_for_rotated_8_2, new_labels_for_rotated_8_3,
			 new_labels_for_rotated_9, new_labels_for_rotated_9_2, new_labels_for_rotated_9_3,
			 new_labels_for_rotated_5, new_labels_for_rotated_5_2), axis=0)

		# ML algorithms works better when learning data is mixed
		EXTENDED_DATA_SET_SIZE = 100000
		mixed_indexes = np.random.permutation(EXTENDED_DATA_SET_SIZE)
		mixed_train_set, mixed_train_labels = new_train_set[mixed_indexes], new_labels[mixed_indexes]

		print('Original data set size =', original_data_set.shape[0])
		print('Extended data set size =', mixed_train_set.shape[0])
		return mixed_train_set, mixed_train_labels

	@staticmethod
	def get_test_images():
		test_images = glob.glob(settings.TEST_IMAGES_DIR + '*')
		return test_images

	def _prepare_data(self, not_prepared_data):
		# Normalizing the RGB codes by dividing it to the max RGB value.
		prepared_data = not_prepared_data / 255

		# used models needs 2 dimensions data
		prepared_data = prepared_data.reshape(prepared_data.shape[0], 784)

		# reduce RAM requirements
		prepared_data = prepared_data.astype(np.float32)

		return prepared_data

	@staticmethod
	def _get_synthesis_data(original_data_set, original_labels, extending_digit, extending_size, operation_type):
		# synthesis of new images by working with array data:
		# available operation:
		# - rotate + zoom
		# - rotate
		# - adding blur
		# - adding noise

		extending_digit_indexes, = np.where(original_labels == extending_digit)

		# mixing filtered data set in order to using different data for different operations
		mixed_indexes = np.random.permutation(extending_digit_indexes)

		filtered_digits_labels = np.take(original_labels, mixed_indexes)
		filtered_digits_set = original_data_set[mixed_indexes, :]

		new_digits_set = filtered_digits_set[0:extending_size]
		new_digits_labels = filtered_digits_labels[0:extending_size]

		default_img_size = 28 # mnist image has 28x28 size
		synthesis_data_set = []
		for x in new_digits_set:
			# for some operations the data must be 28x28
			image = np.reshape(x, (-1, 28))
			new_img = []

			if operation_type == 'rotate_and_zoom':
				# zoom
				zoom_factor = random.choice([0.9, 1, 1, 1.1])
				zoomed_img = []
				# zooming with original size saving
				# the trick from https://stackoverflow.com/questions/37119071/scipy-rotate-and-zoom-an-image-without-changing-its-dimensions
				if zoom_factor < 1:
					delta_img_size = int(np.round(default_img_size * zoom_factor))
					delta_img_move = (default_img_size - default_img_size) // 2
					zoomed_img = np.zeros((default_img_size, default_img_size))
					zoomed_img[delta_img_move:delta_img_move+delta_img_size, delta_img_move:delta_img_move+delta_img_size] = zoom(image, zoom_factor)
				else:
					delta_img_size = int(np.round(default_img_size / zoom_factor))
					delta_img_move = (default_img_size - default_img_size) // 2
					zoomed_img = zoom(image[delta_img_move:delta_img_move+delta_img_size, delta_img_move:delta_img_move+delta_img_size], zoom_factor)
					trim = ((zoomed_img.shape[0] - default_img_size) // 2)
					zoomed_img = zoomed_img[trim:trim+default_img_size, trim:trim+default_img_size]

				# rotate
				# rotation is counterclockwise only because many digits are inclined to the right
				angle = random.choice([15, 17, 18, 19, 20])
				new_img = rotate(zoomed_img, angle, reshape=False)

			if operation_type == 'rotate':
				angle = random.choice([15, 17, 18, 19, 20, 21, 22])
				new_img = rotate(image, angle, reshape=False)

			if operation_type == 'noise':
				noise = np.random.rand(28, 28)
				new_img = image + noise

			if operation_type == 'blur':
				new_img = gaussian_filter(image, 0.4)

			# returning data to original view
			new_img = np.reshape(new_img, 784)
			synthesis_data_set.append(new_img)

		synthesis_data_set = np.asarray(synthesis_data_set)
		return synthesis_data_set, new_digits_labels
