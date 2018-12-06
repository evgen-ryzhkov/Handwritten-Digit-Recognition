from scripts.data import DigitsData
from scripts.model import DigitClassifier


#  --------------
# Step 1
# getting data and familiarity with it

data_obj = DigitsData()
original_train_images, original_train_labels = data_obj.get_train_set()

# visual familiarity with data
# orig_dig = original_train_images[0]
# data_obj.show_image_from_data_set(orig_dig)


# --------------
# Step 2
# model selection and training

# digit_clf_obj = DigitClassifier()
# digit_clf_obj.send_train_set(original_train_images, original_train_labels)

# 2.1. comparing models
# digit_clf_obj.compare_models()

# 2.2. configurating hyperparameters for selected model
# digit_clf_obj.configure_hyperparameters()

# 2.3. getting final model after adding the best parameters to train_final_model method
# final_model = digit_clf_obj.train_final_model()


# --------------
# Step 3
# test model with test set
# if metrics with test data - OK, go to the next step
# else go back to the 2nd step

# test_set_prepared, test_set_labels = data_obj.get_test_set()
# y_pred = final_model.predict(test_set_prepared)
# digit_clf_obj.send_train_set(test_set_prepared, y_pred)
# digit_clf_obj.calculate_model_metrics(test_set_labels, y_pred, 'Test data set prediction:')


# ------------
# Step 4
# improving of model performance

# 4.1. searching the way how to improving of model
# the worst accuracies are for '8' and '9'
# maybe it need to add samples of this digits in train set

# how many do we have this digits in train set
# train_set_size = int(original_train_labels.shape[0])
# nine_digit_quantity = original_train_labels[original_train_labels == 8].shape[0]
# print('Percent of 8 digits in train set = ', nine_digit_quantity / train_set_size * 100, '%')

# 4.2. saving original mnist data as images for further synthesis data creating
# data_obj.save_original_images(train_images)

# 4.3. adding synthesis data to train set
# extended_train_set, extended_labels = data_obj.get_extended_data(original_train_images, original_train_labels)

# looking examples of created images
# data_obj.show_image_from_data_set(extended_train_set[0])
# data_obj.show_image_from_data_set(extended_train_set[60000])
# data_obj.show_image_from_data_set(extended_train_set[1])
# data_obj.show_image_from_data_set(extended_train_set[60001])
# data_obj.show_image_from_data_set(extended_train_set[2])
# data_obj.show_image_from_data_set(extended_train_set[60002])

# checking of matching images and labels
# data_obj.show_image_from_data_set(extended_train_set[100])
# data_obj.show_image_from_data_set(extended_train_set[2000])
# data_obj.show_image_from_data_set(extended_train_set[4100])
# data_obj.show_image_from_data_set(extended_train_set[5000])
# data_obj.show_image_from_data_set(extended_train_set[60000])
# data_obj.show_image_from_data_set(extended_train_set[61000])
# data_obj.show_image_from_data_set(extended_train_set[64500])
# data_obj.show_image_from_data_set(extended_train_set[65900])
# print('Label 1=', extended_labels[100])
# print('Label 1=', extended_labels[2000])
# print('Label 1=', extended_labels[4100])
# print('Label 1=', extended_labels[5000])
# print('Label 1=', extended_labels[60000])
# print('Label 2=', extended_labels[61000])
# print('Label 3=', extended_labels[64500])
# print('Label 4=', extended_labels[65900])

# 4.4. training model with extended data and checking of the new model performance
# del original_train_images, original_train_labels # freeing memory from unused data
#
# digit_clf_obj = DigitClassifier()
# digit_clf_obj.send_train_set(extended_train_set, extended_labels)
# final_model = digit_clf_obj.train_final_model()

# 4.5. if model metrics with train data is ok
# check if it's ok with test data

# test_set_prepared, test_set_labels = data_obj.get_test_set()
# y_pred = final_model.predict(test_set_prepared)
# digit_clf_obj.send_train_set(test_set_prepared, y_pred)
# digit_clf_obj.calculate_model_metrics(test_set_labels, y_pred, 'Test data set prediction:')

# 4.6. saving final model
# digit_clf_obj.save_final_model(final_model)


# ------------
# Step 5
# testing of model with test images (not test data set)
digit_clf_obj = DigitClassifier()
test_images = data_obj.get_test_images()

for test_img in test_images:
	data_obj.show_image_from_file(test_img)
	predicted_digit = digit_clf_obj.classify_by_image(test_img)
	print('Defined digit = ', predicted_digit)
