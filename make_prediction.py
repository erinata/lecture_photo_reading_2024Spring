import pickle
import pandas
from keras.preprocessing.image import ImageDataGenerator

import numpy
import glob

machine = pickle.load(open('cnn_image_machine.pickle', 'rb'))

new_data = ImageDataGenerator(rescale=1./255)
new_data = new_data.flow_from_directory(directory="dataset/new_images",
                            shuffle=False, 
                            target_size=(50,50), 
                            batch_size=1)

new_data.reset()

new_data_length = len([i for i in glob.glob('dataset/new_images/unknown_images/*.jpg')])

prediction = numpy.argmax(machine.predict(new_data, steps=new_data_length), axis=1)
print(prediction)

# print(new_data.filenames)

results = [[new_data.filenames[i], prediction[i]] for i in range(new_data_length)]
results_dataframe = pandas.DataFrame(results, columns=['image', 'prediction'])
results_dataframe.to_csv('predictions.csv', index=False)

