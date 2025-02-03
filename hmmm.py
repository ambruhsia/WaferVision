import streamlit as st
import numpy as np
import pandas as pd
import pickle
from uhm import preprocess_wafer_data
import pickle
from xgboost import Booster
import json 

# Load the model using the old XGBoost version
filename = 'C:/Users/rajpu/atarashi no desu yo/templates/savemodle3.sav'
model = pickle.load(open(filename, 'rb'))

# Ensure the loaded model is a Booster instance
if isinstance(model, Booster):
    # Save the model in XGBoost's JSON format
    model.save_model('saved_model.json')

    # Load the model in XGBoost's JSON format
    model_loaded = Booster()
    model_loaded.load_model('saved_model.json')
else:
    print("The loaded object is not an instance of Booster.")


st.title('WaferMap defect detect :ship:')
 
x = st.text_input("Input fab data (e.g., [[0, 1, 0], [1, 0, 1], [0, 1, 0]])")

def predict():
        
        # Convert input to NumPy array
        wafer_map = np.array(eval(x))  # Convert string to array
        st.write(f"Input converted to array with shape: {wafer_map.shape}")

        # Preprocess the data
        preprocessed_data = preprocess_wafer_data(wafer_map)

        # Predict using the model
        results = model.predict(preprocessed_data)
        st.write(f"Prediction: {results[0]}")
    # except Exception as e:
    #     st.error(f"Error during prediction: {e}")

# Trigger prediction
trigger = st.button('Predict', on_click=predict)
# import warnings
# warnings.filterwarnings("ignore")
 
# df = pd.read_pickle("C:/Users/rajpu/atarashi no desu yo/templates/LSWMD (1).pkl")

# df.info()
 
# df = df.drop(['waferIndex'], axis = 1)
 
# def find_dim(x):
#     dim0=np.size(x,axis=0)
#     dim1=np.size(x,axis=1)
#     return dim0,dim1
# df['waferMapDim']=df.waferMap.apply(find_dim)
 

# uni_waferDim=np.unique(df.waferMapDim, return_counts=True)
 
 
# df['failureNum']=df.failureType
# df['trainTestNum']=df.trianTestLabel


# mapping_type={'Center':0,'Donut':1,'Edge-Loc':2,'Edge-Ring':3,'Loc':4,'Random':5,'Scratch':6,'Near-full':7,'none':8}
# mapping_traintest={'Training':0,'Test':1}

# df = df.replace({'failureType': mapping_type, 'trianTestLabel': mapping_traintest})

 
# tol_wafers = df.shape[0]  
# df_withlabel = df[(df['failureNum']>=0) & (df['failureNum']<=8)]
# df_withlabel =df_withlabel.reset_index()
# df_withpattern = df[(df['failureNum']>=0) & (df['failureNum']<=7)]
# df_withpattern = df_withpattern.reset_index()
# df_nonpattern = df[(df['failureNum']==8)]
# df_withlabel.shape[0], df_withpattern.shape[0], df_nonpattern.shape[0]
 
  
# import skimage
# from skimage import measure
# from skimage.transform import radon
# from skimage.transform import probabilistic_hough_line
# from skimage import measure
# from scipy import interpolate 
# import numpy as np 
# from skimage import measure
# from scipy import stats
 
# def cal_den(x):
#     return 100*(np.sum(x==2)/np.size(x))  

# def find_regions(x):
#     rows=np.size(x,axis=0)
#     cols=np.size(x,axis=1)
#     ind1=np.arange(0,rows,rows//5)
#     ind2=np.arange(0,cols,cols//5)
    
#     reg1=x[ind1[0]:ind1[1],:]
#     reg3=x[ind1[4]:,:]
#     reg4=x[:,ind2[0]:ind2[1]]
#     reg2=x[:,ind2[4]:]

#     reg5=x[ind1[1]:ind1[2],ind2[1]:ind2[2]]
#     reg6=x[ind1[1]:ind1[2],ind2[2]:ind2[3]]
#     reg7=x[ind1[1]:ind1[2],ind2[3]:ind2[4]]
#     reg8=x[ind1[2]:ind1[3],ind2[1]:ind2[2]]
#     reg9=x[ind1[2]:ind1[3],ind2[2]:ind2[3]]
#     reg10=x[ind1[2]:ind1[3],ind2[3]:ind2[4]]
#     reg11=x[ind1[3]:ind1[4],ind2[1]:ind2[2]]
#     reg12=x[ind1[3]:ind1[4],ind2[2]:ind2[3]]
#     reg13=x[ind1[3]:ind1[4],ind2[3]:ind2[4]]
    
#     fea_reg_den = []
#     fea_reg_den = [cal_den(reg1),cal_den(reg2),cal_den(reg3),cal_den(reg4),cal_den(reg5),cal_den(reg6),cal_den(reg7),cal_den(reg8),cal_den(reg9),cal_den(reg10),cal_den(reg11),cal_den(reg12),cal_den(reg13)]
#     return fea_reg_den
 
# df_withpattern['fea_reg']=df_withpattern.waferMap.apply(find_regions)
  
# def change_val(img):
#     img[img==1] =0  
#     return img

# df_withpattern_copy = df_withpattern.copy()
# df_withpattern_copy['new_waferMap'] =df_withpattern_copy.waferMap.apply(change_val)
 
# first_row = df_withpattern_copy['new_waferMap'].tail(1).values[0]
 
# def cubic_inter_mean(img):
#     theta = np.linspace(0., 180., max(img.shape), endpoint=False)
#     sinogram = radon(img, theta=theta)
#     xMean_Row = np.mean(sinogram, axis = 1)
#     x = np.linspace(1, xMean_Row.size, xMean_Row.size)
#     y = xMean_Row
#     f = interpolate.interp1d(x, y, kind = 'cubic')
#     xnew = np.linspace(1, xMean_Row.size, 20)
#     ynew = f(xnew)/100   # use interpolation function returned by `interp1d`
#     return ynew

# def cubic_inter_std(img):
#     theta = np.linspace(0., 180., max(img.shape), endpoint=False)
#     sinogram = radon(img, theta=theta)
#     xStd_Row = np.std(sinogram, axis=1)
#     x = np.linspace(1, xStd_Row.size, xStd_Row.size)
#     y = xStd_Row
#     f = interpolate.interp1d(x, y, kind = 'cubic')
#     xnew = np.linspace(1, xStd_Row.size, 20)
#     ynew = f(xnew)/100   # use interpolation function returned by `interp1d`
#     return ynew  
 
# df_withpattern_copy['fea_cub_mean'] =df_withpattern_copy.waferMap.apply(cubic_inter_mean)
# df_withpattern_copy['fea_cub_std'] =df_withpattern_copy.waferMap.apply(cubic_inter_std) 
# import numpy as np 
# from skimage import measure
# from scipy import stats
 
# from skimage import measure
# from scipy import stats
# import numpy as np

# def cal_dist(img, x, y):
#     dim0 = np.size(img, axis=0)    
#     dim1 = np.size(img, axis=1)
#     dist = np.sqrt((x - dim0 / 2) ** 2 + (y - dim1 / 2) ** 2)
#     return dist  

# def fea_geom(img):
#     norm_area = img.shape[0] * img.shape[1]
#     norm_perimeter = np.sqrt((img.shape[0]) ** 2 + (img.shape[1]) ** 2)
    
#     # Removed 'neighbors' and kept 'connectivity'
#     img_labels = measure.label(img, connectivity=1, background=0)

#     if img_labels.max() == 0:
#         img_labels[img_labels == 0] = 1
#         no_region = 0
#     else:
#         info_region = stats.mode(img_labels[img_labels > 0], axis=None)
#         no_region = info_region.mode[0] - 1  # Updated to access `.mode[0]`
    
#     prop = measure.regionprops(img_labels)
#     prop_area = prop[no_region].area / norm_area
#     prop_perimeter = prop[no_region].perimeter / norm_perimeter 
    
#     prop_cent = prop[no_region].local_centroid 
#     prop_cent = cal_dist(img, prop_cent[0], prop_cent[1])
    
#     prop_majaxis = prop[no_region].major_axis_length / norm_perimeter 
#     prop_minaxis = prop[no_region].minor_axis_length / norm_perimeter  
#     prop_ecc = prop[no_region].eccentricity  
#     prop_solidity = prop[no_region].solidity  
    
#     return prop_area, prop_perimeter, prop_majaxis, prop_minaxis, prop_ecc, prop_solidity

# # Apply the function to the DataFrame
# df_withpattern_copy['fea_geom'] = df_withpattern_copy['waferMap'].apply(fea_geom)
 
# df_withpattern_copy.fea_geom[340] #donut
 
# df_all=df_withpattern_copy.copy()
# a=[df_all.fea_reg[i] for i in range(df_all.shape[0])] #13
# b=[df_all.fea_cub_mean[i] for i in range(df_all.shape[0])] #20
# c=[df_all.fea_cub_std[i] for i in range(df_all.shape[0])] #20
# d=[df_all.fea_geom[i] for i in range(df_all.shape[0])] #6
# fea_all = np.concatenate((np.array(a),np.array(b),np.array(c),np.array(d)),axis=1) #59 in total
 
# label=[df_all.failureNum[i] for i in range(df_all.shape[0])]
# label=np.array(label)

# # %% [markdown]
# # **Step3: Choose algorithms**

# # %% [markdown]
# # * If you have no idea which algorithm to choose, you may have a look on this Microsoft Azure Machine Learning Algorithm Cheat Sheet. Here is the link:[Machine Learning Algorithm Cheat Sheet](https://unsupervisedmethods.com/cheat-sheet-of-machine-learning-and-python-and-math-cheat-sheets-a4afe4e791b6)

# # %% [markdown]
# # > No Best Machine Learning Algorithm
# # 
# # You cannot know a priori which algorithm will be best suited for your problem.
# # 
# # Here are some tips:
# # 
# # * You can apply your favorite algorithm.
# # * You can apply the algorithm recommended in a book or paper.
# # * You can apply the algorithm that is winning the most Kaggle competitions right now.
# # * You can apply the algorithm that works best with your test rig, infrastructure, database, or whatever.
# # 
# # For our multi class classification problem, we choose the most popular SVMs at the moment. 

# # %% [markdown]
# # 
# # > This module implements multiclass and multilabel learning algorithms: 
# # 
# # Refer to [scikits learning](http://ogrisel.github.io/scikit-learn.org/sklearn-tutorial/modules/multiclass.html)
# # 
# # * **one-vs-the-rest :** one-vs-the-rest strategy consists in fitting one classifier per class. For each classifier, the class is fitted against all the other classes. 
# # 
# # * **one-vs-one:** one-vs-one classifier constructs one classifier per pair of classes.
# # 
# # * error correcting output codes
# # 
# # We choose **One-VS-One multi-class SVMs** as our model based on literature review for this dataset.
# # 

# # %%

# from sklearn.model_selection import train_test_split
  

# X = fea_all
# y = label

# from collections import  Counter
# X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)                      
# print('Training target statistics: {}'.format(Counter(y_train)))
# print('Testing target statistics: {}'.format(Counter(y_test)))

# RANDOM_STATE =42


# # %%
# # ---multicalss classification ---# 
# # One-Vs-One
# from sklearn.svm import LinearSVC
# from sklearn.multiclass import OneVsOneClassifier
# clf2 = OneVsOneClassifier(LinearSVC(random_state = RANDOM_STATE)).fit(X_train, y_train)
# y_train_pred = clf2.predict(X_train)
# y_test_pred = clf2.predict(X_test)
# train_acc2 = np.sum(y_train == y_train_pred, axis=0, dtype='float') / X_train.shape[0]
# test_acc2 = np.sum(y_test == y_test_pred, axis=0, dtype='float') / X_test.shape[0]
# print('One-Vs-One Training acc: {}'.format(train_acc2*100)) #One-Vs-One Training acc: 80.36
# print('One-Vs-One Testing acc: {}'.format(test_acc2*100)) #One-Vs-One Testing acc: 79.04
# print("y_train_pred[:100]: ", y_train_pred[:100])
# print ("y_train[:100]: ", y_train[:100])

# # %% [markdown]
# # * The overall training accuracy is: **80.36%**
# # 
# # * The overall testing accuracy is: **79.04%**

# # %% [markdown]
# # **Step4: Present results**

# # %% [markdown]
# # * pattern recognition confusion matrix
 
# from sklearn.ensemble import RandomForestClassifier
# clf_rf = RandomForestClassifier(random_state=RANDOM_STATE).fit(X_train, y_train)
# y_test_pred_rf = clf_rf.predict(X_test)
# print('Random Forest Test acc: {}'.format((y_test == y_test_pred_rf).mean() * 100))

 

# from sklearn.model_selection import GridSearchCV
# param_grid = {'C': [0.01, 0.1, 1, 10]}
# grid_search = GridSearchCV(LinearSVC(random_state=RANDOM_STATE), param_grid, cv=5)
# grid_search.fit(X_train, y_train)
# print("Best parameters:", grid_search.best_params_)

 
# from xgboost import XGBClassifier

# clf_xgb = XGBClassifier(random_state=RANDOM_STATE, use_label_encoder=False, eval_metric='mlogloss')
# clf_xgb.fit(X_train, y_train)
# y_test_pred_xgb = clf_xgb.predict(X_test)

# print('XGBoost Test Accuracy: {:.2f}%'.format((y_test == y_test_pred_xgb).mean() * 100))
 
# from lightgbm import LGBMClassifier

# clf_lgbm = LGBMClassifier(random_state=RANDOM_STATE)
# clf_lgbm.fit(X_train, y_train)
# y_test_pred_lgbm = clf_lgbm.predict(X_test)

# print('LightGBM Test Accuracy: {:.2f}%'.format((y_test == y_test_pred_lgbm).mean() * 100))
 
# class_labels = np.unique(y_test)

  
# # from sklearn.neighbors import KNeighborsClassifier

# # clf_knn = KNeighborsClassifier(n_neighbors=5)
# # clf_knn.fit(X_train, y_train)
# # y_test_pred_knn = clf_knn.predict(X_test)

# # print('k-NN Test Accuracy: {:.2f}%'.format((y_test == y_test_pred_knn).mean() * 100))


# # %%
# # from sklearn.naive_bayes import MultinomialNB

# # clf_nb = MultinomialNB()
# # clf_nb.fit(X_train, y_train)
# # y_test_pred_nb = clf_nb.predict(X_test)

# # print('Naive Bayes Test Accuracy: {:.2f}%'.format((y_test == y_test_pred_nb).mean() * 100))

 
# from sklearn.svm import SVC

# clf_svc = SVC(kernel='rbf', random_state=RANDOM_STATE)
# clf_svc.fit(X_train, y_train)
# y_test_pred_svc = clf_svc.predict(X_test)

# print('SVM with RBF Kernel Test Accuracy: {:.2f}%'.format((y_test == y_test_pred_svc).mean() * 100))
 
# # from keras.models import Sequential
# # from keras.layers import Dense
# # num_classes = len(set(y_train))

# # model = Sequential([
# #     Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
# #     Dense(64, activation='relu'),
# #     Dense(num_classes, activation='softmax')  # Ensure num_classes is defined
# # ])

# # model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# # model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10)

 
# import pickle
# # filename1 = 'savemodle1.sav'
# # pickle.dump(clf_lgbm, open(filename1, 'wb'))


# # filename2 = 'savemodle2.sav'
# # pickle.dump(clf_rf, open(filename2, 'wb'))


# filename3 = 'savemodle3.sav'
# pickle.dump(clf_xgb, open(filename3, 'wb'))


# # filename4 = 'savemodle4.sav'
# # pickle.dump(clf_knn, open(filename4, 'wb'))



# # %%

# def preprocess_wafer_data(x):
#     """
#     Preprocess wafer map data and return the feature matrix X.

#     Parameters:
#         x: List of wafer map arrays

#     Returns:
#         X: Numpy array of processed features
#     """

#     def cal_den(region):
#         "Calculate density of regions with value 2."
#         return 100 * (np.sum(region == 2) / np.size(region))

#     def find_regions(wafer_map):
#         "Divide the wafer map into regions and calculate density for each region."
#         rows, cols = wafer_map.shape
#         ind1 = np.arange(0, rows, rows // 5)
#         ind2 = np.arange(0, cols, cols // 5)

#         reg1 = wafer_map[ind1[0]:ind1[1], :]
#         reg3 = wafer_map[ind1[4]:, :]
#         reg4 = wafer_map[:, ind2[0]:ind2[1]]
#         reg2 = wafer_map[:, ind2[4]:]

#         reg5 = wafer_map[ind1[1]:ind1[2], ind2[1]:ind2[2]]
#         reg6 = wafer_map[ind1[1]:ind1[2], ind2[2]:ind2[3]]
#         reg7 = wafer_map[ind1[1]:ind1[2], ind2[3]:ind2[4]]
#         reg8 = wafer_map[ind1[2]:ind1[3], ind2[1]:ind2[2]]
#         reg9 = wafer_map[ind1[2]:ind1[3], ind2[2]:ind2[3]]
#         reg10 = wafer_map[ind1[2]:ind1[3], ind2[3]:ind2[4]]
#         reg11 = wafer_map[ind1[3]:ind1[4], ind2[1]:ind2[2]]
#         reg12 = wafer_map[ind1[3]:ind1[4], ind2[2]:ind2[3]]
#         reg13 = wafer_map[ind1[3]:ind1[4], ind2[3]:ind2[4]]

#         fea_reg_den = [
#             cal_den(reg1), cal_den(reg2), cal_den(reg3), cal_den(reg4),
#             cal_den(reg5), cal_den(reg6), cal_den(reg7), cal_den(reg8),
#             cal_den(reg9), cal_den(reg10), cal_den(reg11), cal_den(reg12), cal_den(reg13)
#         ]
#         return fea_reg_den

#     def change_val(wafer_map):
#         "Replace all values of 1 in the wafer map with 0."
#         wafer_map[wafer_map == 1] = 0
#         return wafer_map

#     def cubic_inter_mean(wafer_map):
#         "Perform cubic interpolation on the mean row of the sinogram."
#         theta = np.linspace(0., 180., max(wafer_map.shape), endpoint=False)
#         sinogram = radon(wafer_map, theta=theta)
#         x_mean_row = np.mean(sinogram, axis=1)
#         x = np.linspace(1, x_mean_row.size, x_mean_row.size)
#         y = x_mean_row
#         f = interpolate.interp1d(x, y, kind='cubic')
#         xnew = np.linspace(1, x_mean_row.size, 20)
#         ynew = f(xnew) / 100
#         return ynew

#     def cubic_inter_std(wafer_map):
#         "Perform cubic interpolation on the standard deviation of the sinogram."
#         theta = np.linspace(0., 180., max(wafer_map.shape), endpoint=False)
#         sinogram = radon(wafer_map, theta=theta)
#         x_std_row = np.std(sinogram, axis=1)
#         x = np.linspace(1, x_std_row.size, x_std_row.size)
#         y = x_std_row
#         f = interpolate.interp1d(x, y, kind='cubic')
#         xnew = np.linspace(1, x_std_row.size, 20)
#         ynew = f(xnew) / 100
#         return ynew

#     def cal_dist(wafer_map, x, y):
#         "Calculate the distance of a point from the center of the wafer map."
#         dim0, dim1 = wafer_map.shape
#         dist = np.sqrt((x - dim0 / 2) ** 2 + (y - dim1 / 2) ** 2)
#         return dist

#     def fea_geom(wafer_map):
#         "Extract geometric features from the largest region in the wafer map."
#         norm_area = wafer_map.shape[0] * wafer_map.shape[1]
#         norm_perimeter = np.sqrt((wafer_map.shape[0]) ** 2 + (wafer_map.shape[1]) ** 2)

#         img_labels = measure.label(wafer_map, connectivity=1, background=0)
#         no_region = img_labels.max() - 1 if img_labels.max() > 0 else 0

#         prop = measure.regionprops(img_labels)
#         if no_region < len(prop):
#             prop_area = prop[no_region].area / norm_area
#             prop_perimeter = prop[no_region].perimeter / norm_perimeter
#             prop_centroid = cal_dist(wafer_map, *prop[no_region].local_centroid)
#             prop_majaxis = prop[no_region].major_axis_length / norm_perimeter
#             prop_minaxis = prop[no_region].minor_axis_length / norm_perimeter
#             prop_eccentricity = prop[no_region].eccentricity
#             prop_solidity = prop[no_region].solidity
#         else:
#             # Fallback for edge cases
#             prop_area, prop_perimeter, prop_majaxis, prop_minaxis = 0, 0, 0, 0
#             prop_eccentricity, prop_solidity, prop_centroid = 0, 0, 0

#         return prop_area, prop_perimeter, prop_majaxis, prop_minaxis, prop_eccentricity, prop_solidity

#     # Preprocessing pipeline
#     fea_reg = [find_regions(x)  ] 
#     fea_cub_mean = [cubic_inter_mean(x)    ]
#     fea_cub_std = [cubic_inter_std(x)  ]
#     fea_geom = [fea_geom(x)  ]

#     # Combine all features into a single array
#     a = np.array(fea_reg)
#     b = np.array(fea_cub_mean)
#     c = np.array(fea_cub_std)
#     d = np.array(fea_geom)

#     X = np.concatenate((a, b, c, d), axis=1)
#     return X

  
# load_model=pickle.load(open(filename3, 'rb'))

# # load_model.predict(processed_data)
 
# import pickle
# from xgboost import XGBClassifier, Booster

# # Define Random State
# RANDOM_STATE = 42

# # Train the XGBClassifier model
# clf_xgb = XGBClassifier(random_state=RANDOM_STATE, use_label_encoder=False, eval_metric='mlogloss')
# clf_xgb.fit(X_train, y_train)

# # Test prediction and accuracy
# y_test_pred_xgb = clf_xgb.predict(X_test)
# print('XGBoost Test Accuracy: {:.2f}%'.format((y_test == y_test_pred_xgb).mean() * 100))

# # Save the model using pickle
# filename3 = 'saved_model_xgb.sav'
# pickle.dump(clf_xgb, open(filename3, 'wb'))

# # Save the model as JSON using Booster
# booster = clf_xgb.get_booster()
# booster.save_model('xgb_model.json')  # Save as JSON

# # Load the model back using Booster
# loaded_booster = Booster()
# loaded_booster.load_model('xgb_model.json')  # Load the JSON model

# # Optional: Using the loaded booster for predictions
# # You'd need to pass the data in DMatrix format for predictions
# import xgboost as xgb
# dtest = xgb.DMatrix(X_test)  # Convert X_test to DMatrix
# y_loaded_pred = loaded_booster.predict(dtest)

# # If required, print the predictions from the loaded model
# print("Predictions from loaded Booster model:", y_loaded_pred)

 