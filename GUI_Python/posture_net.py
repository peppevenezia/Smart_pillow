#!./bin/python3.9
#%%
import datetime as dt
from distutils.command.install_egg_info import to_filename
from weakref import finalize
import joblib
import pandas as pd 
import h5py
import platform
from keras.models import load_model
import numpy as np
from Include.custom_scaler import custom_MinMaxScaler
from os.path import isdir, isfile, join
from os import listdir, makedirs
from sklearn.pipeline import Pipeline
import tensorflow as tf
import keras
import tensorflow_addons as tfa
from keras.wrappers.scikit_learn import KerasClassifier
import itertools as it
from keras.callbacks import ModelCheckpoint,ReduceLROnPlateau,EarlyStopping
import visualkeras

pixelX=45
pixelY=41
finestra = 1 #hyperparameter (frames considered per prediction)
if platform.system()=="Windows":
    str_path = "..\Data\RAW" 
    dir_list = [str_path+"\\"+ f for f in listdir(str_path) if isdir(join(str_path, f))] #files for each user
else:
    str_path = "../Data/RAW"
    dir_list = [str_path+"/"+ f for f in listdir(str_path) if isdir(join(str_path, f))] 


dir_list = sorted(dir_list)
dir_list_train=[]
dir_list_val=[]
dir_list_test=[]

#split in stratified way mainly bases on height and sex (calibration should be invariant to weight)
#different users are in differents sets since testing on a user used for training would provide an 
#optimistic estimate with the actual performance on new unseen data
tr_data=[0,1,3,4,6,7,9,10,12,13] 
v_data=[5,8,14]
t_data=[2,11]

lists=[dir_list_train,dir_list_val,dir_list_test]
datas=[tr_data,v_data,t_data]

for i in range(len(lists)):
    for patient in datas[i]:
        lists[i].append(dir_list[patient]) 


file_list_train= []
file_list_val= []
file_list_test= []

#data per user per set
for i in dir_list_train:
    if platform.system()=="Windows":
        file_list_train.append( [i+"\\" + f for f in listdir(i)  if isfile(join(i,f))]) #to save all the path of the acquisition of the user 
    else:
        file_list_train.append( [i+"/" + f for f in listdir(i)  if isfile(join(i,f))]) #to save all the path of the acquisition of the user 

for i in dir_list_val:
    if platform.system()=="Windows":
        file_list_val.append( [i+"\\" + f for f in listdir(i)  if isfile(join(i,f))]) #to save all the path of the acquisition of the user 
    else:
        file_list_val.append( [i+"/" + f for f in listdir(i)  if isfile(join(i,f))]) #to save all the path of the acquisition of the user 

for i in dir_list_test: 
    if platform.system()=="Windows":
        file_list_test.append( [i+"\\" + f for f in listdir(i)  if isfile(join(i,f))]) #to save all the path of the acquisition of the user 
    else:
        file_list_test.append( [i+"/" + f for f in listdir(i)  if isfile(join(i,f))]) #to save all the path of the acquisition of the user 


#%%
def read_data(file_list):
    dataset=[]
    dataframe=np.empty((0,pixelX,pixelY),int)
    for User in file_list:
        for file in User: #data from each user
            h5= h5py.File(file,'r')
            data = h5.get('image')
            data = np.array(data)
            data=np.delete(data,np.arange(0,15),axis=0) #remove the first 15 values during which calibration occures 
            dataset.append(data)
            dataframe =np.append(dataframe,data,axis=0) #here all the data are stored all togheter to one dataframe 
    return dataframe,dataset



#create dataset for each set
X_train,dataset_train = read_data(file_list_train)
X_val,dataset_val = read_data(file_list_val)
X_test,dataset_test = read_data(file_list_test)

#target creation
def target_creation(dataset,x): #using dataset
    y=[] 
    for Acquisition in range(len(dataset)): #data is acquired from each user in standardized way: 1 min normal posture, 1 min forwards, 1 min backwards, 1 min right and 1 min left (after calibration)
        for row in range(len(dataset[Acquisition])):
            if row<=200:
                y.append(0)
            if row>200 and row<=400:
                y.append(1) # gam
            if row>400 and row<=600:
                y.append(2)
            if row>600 and row<=800:
                y.append(3)
            if row>800 and row<=1000:
                y.append(4)
            if row>1000:
                y.append(4)

    return np.array(y)

y_train = target_creation(dataset_train,X_train)
y_val = target_creation(dataset_val,X_val)
y_test = target_creation(dataset_test,X_test)

from sklearn.utils import class_weight #this was initially used when more data related to normal posture was present but it is kept all the same (it accounts for small data imbalance anyway)
class_weights = class_weight.compute_class_weight(
            class_weight='balanced', 
            classes=np.unique(y_train),
            y=y_train)
class_weights = dict(enumerate(class_weights))


#%%
#convolutional architecture was seen to perform worse since the most relevant information to predict the posture is the baricenter position 
#rather than local correlations in data: for this reason we passed from this more complex convolutional network to a simpler but better performing architecture (below)
'''
def create_posture_model(image_heigth=pixelX,image_width=pixelY, image_depth=3,image_value=1,learning_rate=0.001):

    input_layer = keras.layers.Input(shape=[image_heigth,image_width, image_depth,image_value])
    layer_1a = keras.layers.Conv3D(16, kernel_size=(3,3,1), activation = 'relu', padding = 'same', kernel_regularizer="l2",kernel_initializer = "he_normal")(input_layer) #1 with 3
    layer_1b = keras.layers.Conv3D(16, kernel_size=(5,5,1), activation = 'relu', padding = 'same', kernel_regularizer="l2",kernel_initializer = "he_normal")(input_layer) #1 with 3
    layer_1c = keras.layers.Conv3D(16, kernel_size=(7,7,2), activation = 'relu', padding = 'same', kernel_regularizer="l2",kernel_initializer = "he_normal")(input_layer) #1 with 3
    
    pool_1a = keras.layers.MaxPooling3D((2,2,1))(layer_1a)
    pool_1b = keras.layers.MaxPooling3D((2,2,1))(layer_1b)
    pool_1c = keras.layers.MaxPooling3D((2,2,1))(layer_1c)

    layer_2a = keras.layers.Conv3D(16, kernel_size=(3,3,1), activation = 'relu', padding = 'same', kernel_regularizer="l2",kernel_initializer = "he_normal")(pool_1a) #1 with 3
    layer_2b = keras.layers.Conv3D(16, kernel_size=(4,4,1), activation = 'relu', padding = 'same', kernel_regularizer="l2",kernel_initializer = "he_normal")(pool_1b) #1 with 3
    layer_2c = keras.layers.Conv3D(16, kernel_size=(5,5,2), activation = 'relu', padding = 'same', kernel_regularizer="l2",kernel_initializer = "he_normal")(pool_1c) #1 with 3

    pool_2a = keras.layers.MaxPooling3D((2,2,1))(layer_2a)
    pool_2b = keras.layers.MaxPooling3D((2,2,1))(layer_2b)
    pool_2c = keras.layers.MaxPooling3D((2,2,1))(layer_2c)

    layer_1 = keras.layers.Concatenate()([pool_2a,pool_2b,pool_2c])
    #layer_1 = keras.layers.BatchNormalization()(layer_1)
    #layer_1 = keras.layers.MaxPooling3D((2,2,1))(layer_1)  
    layer_1 = keras.layers.Dropout(0.5)(layer_1)
    layer_2 = keras.layers.Conv3D(32, kernel_size=(3,3,2), activation = 'relu', padding = 'same', kernel_initializer = "he_normal",kernel_regularizer="l2")(layer_1) #3 with 3
    
    #layer_2 = keras.layers.Dropout(0.5)(layer_2)
    layer_2 = keras.layers.MaxPooling3D()(layer_2)  
    layer_2 = keras.layers.Dropout(0.5)(layer_2)
    
    layer_3 = keras.layers.GlobalAveragePooling3D()(layer_2)
    layer_4 = keras.layers.Dense(32, activation = 'relu', kernel_initializer = "he_normal",kernel_regularizer="l2")(layer_3)
    #layer_5 = keras.layers.Dense(54, activation = 'relu', kernel_initializer = "he_normal")(layer_4)
    layer_4 = keras.layers.Dropout(0.5)(layer_4)
    output_layer = keras.layers.Dense(5, activation = 'softmax', kernel_initializer = "he_normal",kernel_regularizer="l2")(layer_4)
    
    model = keras.Model(inputs=input_layer,outputs= output_layer,name="intention_model")
    f1 = tfa.metrics.F1Score(num_classes = 5, average = 'weighted')
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer = optimizer,loss = 'sparse_categorical_crossentropy', metrics = ["accuracy"]) 
    
    return model
'''

def create_posture_model(image_heigth=pixelY,image_width=pixelX, image_depth=finestra,learning_rate=0.001): #leaning back is getting confused with normal

    input_layer = keras.layers.Input(shape=[image_heigth,image_width, image_depth]) 
    
    pool_1 = keras.layers.AveragePooling2D((2,2))(input_layer) #20,22,1 downsample to decrease resolution (extract average per region (weight distribution is a global feature))

    pool_2 = keras.layers.AveragePooling2D((2,2))(pool_1) #10,11,1 decrease resolution even more


    flat = keras.layers.Flatten()(pool_2) 

    layer_4 = keras.layers.Dense(100, activation = keras.layers.LeakyReLU(0.01), kernel_initializer = "he_normal",kernel_regularizer="l2")(flat) 

    layer_6 = keras.layers.Dense(50, activation = keras.layers.LeakyReLU(0.01), kernel_initializer = "he_normal",kernel_regularizer="l2")(layer_4)
    
    #adding dense layers causes overfitting (too complex model for the little data available)
    output_layer = keras.layers.Dense(5, activation = 'softmax', kernel_initializer = "he_normal",kernel_regularizer="l2")(layer_6)
    
    model = keras.Model(inputs=input_layer,outputs= output_layer,name="posture_net")
    f1 = keras.metrics.F1Score(num_classes = 5, average = 'weighted')
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer = optimizer,loss = 'sparse_categorical_crossentropy', metrics = ["accuracy"]) 
    
    return model



#%%
model = create_posture_model()
print(model.summary())
#tf.keras.utils.plot_model(model, expand_nested=True, to_file= "../cnn.png")
#visualkeras.layered_view(model, legend=True, spacing=20, scale_xy=5, to_file="../cnn.png")

#necessary only for the old acquired data since the shape was transposed; from now on it is acquired correctly: each frame is 41,45
X_train= X_train.reshape(X_train.shape[0],41,45,1)
X_val= X_val.reshape(X_val.shape[0],41,45,1)
X_test= X_test.reshape(X_test.shape[0],41,45,1)


#for sklearn - keras interoperability
posture_model = KerasClassifier(build_fn = create_posture_model, verbose = 1, epochs = 100, batch_size = 100)

#pipeline is used to apply the scaler  and to load the model
pipe = Pipeline(steps=[('scaler', custom_MinMaxScaler(window=finestra)), ('clf', posture_model)])

scaler = custom_MinMaxScaler(window=finestra)
scaler.fit(X_train) #parameters for scaling fit on training set to scale the validation (not in pipeline)
X_val = scaler.transform(X_val)
now = dt.datetime.now() #for checkpoints unicity

#Dir checkpoint
if platform.system()=='Windows':
    makedirs('..\checkpoint\\'+'posture_model' +now.strftime("%a_%d-%m-%Y_%H_%M_%S"))
    path='..\checkpoint\\'+'posture_model' + now.strftime("%a_%d-%m-%Y_%H_%M_%S")
else:
    makedirs('../checkpoint/'+'posture_model' +now.strftime("%a_%d-%m-%Y_%H_%M_%S"))
    path='../checkpoint/'+'posture_model' + now.strftime("%a_%d-%m-%Y_%H_%M_%S")


#avoid overfitting with early stopping (on validation accuracy) and reduce learning rate on plateau to improve training stability
callbacks=[ModelCheckpoint(path,monitor='val_accuracy',mode='max',save_best_only=True),ReduceLROnPlateau('val_accuracy',patience=8,mode='max'),EarlyStopping('val_accuracy',mode='max',patience=8,restore_best_weights=True)]

#when calling .fit method on the pipeline all the .fit methods of the objects within it are called: scaler and model
result = pipe.fit(X_train, y_train,clf__validation_data=((X_val,y_val)),clf__callbacks=callbacks, clf__class_weight=class_weights)





#%%
y_fit=pipe.predict(X_test) #at this point all the .transform of the objects in the pipeline are called (scaler and model (.predict))
#%%




#dir pipeline (to save the trained model)
if platform.system()=='Windows':

    makedirs('..\pipeline\\'+'posture_model' +now.strftime("%a_%d-%m-%Y_%H_%M_%S"))
    path='..\pipeline\\'+'posture_model' + now.strftime("%a_%d-%m-%Y_%H_%M_%S")
else:
    makedirs('../pipeline/'+'posture_model' +now.strftime("%a_%d-%m-%Y_%H_%M_%S"))
    path='../pipeline/'+'posture_model' + now.strftime("%a_%d-%m-%Y_%H_%M_%S")
def save_pipeline(pipeline, folder_name):
    if platform.system()=='Windows':
        joblib.dump(pipeline.named_steps["scaler"], open(folder_name+'\\'+'scaler.pkl', 'wb'))
        joblib.dump(pipeline.named_steps['clf'].classes_, open(folder_name+'\\'+'classes.pkl', 'wb'))
        pipeline.named_steps['clf'].model.save(folder_name+'\model.h5')
    else:
        joblib.dump(pipeline.named_steps["scaler"], open(folder_name+'/'+'scaler.pkl', 'wb'))
        joblib.dump(pipeline.named_steps['clf'].classes_, open(folder_name+'/'+'classes.pkl', 'wb'))
        pipeline.named_steps['clf'].model.save(folder_name+'/model.h5')
save_pipeline(pipe, path)











# %%
#metrics (for model evaluation)
import sklearn.metrics as skmetrics
print("\n\033[1;35m Test-set results\033[0m")
acc = skmetrics.accuracy_score(y_test, y_fit)
print("\033[1;33m Posture Model validation accuracy\033[33m: {:<2.2f} %\033[0m".format(100*acc))
np.set_printoptions(precision=4, suppress=True)
print(" \033[1mPrecision:\033[0m", end=' ')
print(skmetrics.precision_score(y_test, y_fit, average=None))
print(" \033[1mRecall:   \033[0m", end=' ')
print(skmetrics.recall_score(y_test, y_fit, average=None))
print(" \033[1mAccuracy: \033[0m", end=' ')
print(skmetrics.accuracy_score(y_test, y_fit))
print(" \033[1mBalanced Accuracy: \033[0m", end=' ')
print(skmetrics.balanced_accuracy_score(y_test, y_fit))
print(" \033[1mF1 score: \033[0m", end=' ')
print(skmetrics.f1_score(y_test, y_fit, average=None))
# %%
