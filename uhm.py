from flask import Flask, render_template, request
import pickle
import numpy as np
from xgboost import XGBClassifier, Booster

# loading libraries
import skimage
from skimage import measure
from skimage.transform import radon
from skimage.transform import probabilistic_hough_line
from skimage import measure
from scipy import interpolate
from scipy import stats
 


def preprocess_wafer_data(x):
    """
    Preprocess wafer map data and return the feature matrix X.

    Parameters:
        x: List of wafer map arrays

    Returns:
        X: Numpy array of processed features
    """

    def cal_den(region):
        "Calculate density of regions with value 2."
        return 100 * (np.sum(region == 2) / np.size(region))

    def find_regions(wafer_map):
        "Divide the wafer map into regions and calculate density for each region."
        # wafer_map = np.array(eval(wafer_map))  # Convert string to array
        rows, cols = wafer_map.shape
        ind1 = np.arange(0, rows, rows // 5)
        ind2 = np.arange(0, cols, cols // 5)

        reg1 = wafer_map[ind1[0]:ind1[1], :]
        reg3 = wafer_map[ind1[4]:, :]
        reg4 = wafer_map[:, ind2[0]:ind2[1]]
        reg2 = wafer_map[:, ind2[4]:]

        reg5 = wafer_map[ind1[1]:ind1[2], ind2[1]:ind2[2]]
        reg6 = wafer_map[ind1[1]:ind1[2], ind2[2]:ind2[3]]
        reg7 = wafer_map[ind1[1]:ind1[2], ind2[3]:ind2[4]]
        reg8 = wafer_map[ind1[2]:ind1[3], ind2[1]:ind2[2]]
        reg9 = wafer_map[ind1[2]:ind1[3], ind2[2]:ind2[3]]
        reg10 = wafer_map[ind1[2]:ind1[3], ind2[3]:ind2[4]]
        reg11 = wafer_map[ind1[3]:ind1[4], ind2[1]:ind2[2]]
        reg12 = wafer_map[ind1[3]:ind1[4], ind2[2]:ind2[3]]
        reg13 = wafer_map[ind1[3]:ind1[4], ind2[3]:ind2[4]]

        fea_reg_den = [
            cal_den(reg1), cal_den(reg2), cal_den(reg3), cal_den(reg4),
            cal_den(reg5), cal_den(reg6), cal_den(reg7), cal_den(reg8),
            cal_den(reg9), cal_den(reg10), cal_den(reg11), cal_den(reg12), cal_den(reg13)
        ]
        return fea_reg_den

    def change_val(wafer_map):
        "Replace all values of 1 in the wafer map with 0."
        wafer_map[wafer_map == 1] = 0
        return wafer_map

    def cubic_inter_mean(wafer_map):
        "Perform cubic interpolation on the mean row of the sinogram."
        theta = np.linspace(0., 180., max(wafer_map.shape), endpoint=False)
        sinogram = radon(wafer_map, theta=theta)
        x_mean_row = np.mean(sinogram, axis=1)
        x = np.linspace(1, x_mean_row.size, x_mean_row.size)
        y = x_mean_row
        f = interpolate.interp1d(x, y, kind='cubic')
        xnew = np.linspace(1, x_mean_row.size, 20)
        ynew = f(xnew) / 100
        return ynew

    def cubic_inter_std(wafer_map):
        "Perform cubic interpolation on the standard deviation of the sinogram."
        theta = np.linspace(0., 180., max(wafer_map.shape), endpoint=False)
        sinogram = radon(wafer_map, theta=theta)
        x_std_row = np.std(sinogram, axis=1)
        x = np.linspace(1, x_std_row.size, x_std_row.size)
        y = x_std_row
        f = interpolate.interp1d(x, y, kind='cubic')
        xnew = np.linspace(1, x_std_row.size, 20)
        ynew = f(xnew) / 100
        return ynew

    def cal_dist(wafer_map, x, y):
        "Calculate the distance of a point from the center of the wafer map."
        dim0, dim1 = wafer_map.shape
        dist = np.sqrt((x - dim0 / 2) ** 2 + (y - dim1 / 2) ** 2)
        return dist

    def fea_geom(wafer_map):
        "Extract geometric features from the largest region in the wafer map."
        norm_area = wafer_map.shape[0] * wafer_map.shape[1]
        norm_perimeter = np.sqrt((wafer_map.shape[0]) ** 2 + (wafer_map.shape[1]) ** 2)

        img_labels = measure.label(wafer_map, connectivity=1, background=0)
        no_region = img_labels.max() - 1 if img_labels.max() > 0 else 0

        prop = measure.regionprops(img_labels)
        if no_region < len(prop):
            prop_area = prop[no_region].area / norm_area
            prop_perimeter = prop[no_region].perimeter / norm_perimeter
            prop_centroid = cal_dist(wafer_map, *prop[no_region].local_centroid)
            prop_majaxis = prop[no_region].major_axis_length / norm_perimeter
            prop_minaxis = prop[no_region].minor_axis_length / norm_perimeter
            prop_eccentricity = prop[no_region].eccentricity
            prop_solidity = prop[no_region].solidity
        else:
            # Fallback for edge cases
            prop_area, prop_perimeter, prop_majaxis, prop_minaxis = 0, 0, 0, 0
            prop_eccentricity, prop_solidity, prop_centroid = 0, 0, 0

        return prop_area, prop_perimeter, prop_majaxis, prop_minaxis, prop_eccentricity, prop_solidity

    # Preprocessing pipeline
    fea_reg = [find_regions(x)  ] 
    fea_cub_mean = [cubic_inter_mean(x)    ]
    fea_cub_std = [cubic_inter_std(x)  ]
    fea_geom = [fea_geom(x)  ]

    # Combine all features into a single array
    a = np.array(fea_reg)
    b = np.array(fea_cub_mean)
    c = np.array(fea_cub_std)
    d = np.array(fea_geom)

    X = np.concatenate((a, b, c, d), axis=1)
    return X


app = Flask(__name__, template_folder='templates')

# import xgboost as xgb

# # Load the model using the old version
# model = xgb.Booster()
# model.load_model('savemodle3.sav')

# # Save it in a compatible format
# model.save_model('xgb_model.json')

from xgboost import Booster

# booster = Booster()
# booster.load_model('savemodle3.sav')


# Load the pickle model

filename3 = r'C:\Users\rajpu\atarashi no desu yo\templates\savemodle3.sav'

load_model = pickle.load(open(filename3, 'rb'))

# # Load the Booster model (JSON format)
# booster = Booster()
# booster.load_model('xgb_model.json')  # Ensure the model has been saved earlier in JSON format

@app.route('/')
def home():
    result = ''
    return render_template('ermm.html', **locals())

@app.route('/predict', methods=['POST', 'GET'])
def predict():
    x = np.array(request.form['x'], dtype=float)  # Assuming input is numeric
    processed_data = preprocess_wafer_data(x)  # Preprocess function to prepare input data

    # Predict using the pickle model
    result_pickle = load_model.predict(processed_data)[0]

    # Predict using the Booster model
    from xgboost import DMatrix
    processed_dmatrix = DMatrix(processed_data)
    result_booster = load_model.predict(processed_dmatrix)

    # Example: Combine results or choose one
    result = f"Pickle Model Result: {result_pickle}, Booster Model Result: {result_booster[0]}"
    return render_template('ermm.html', **locals())

if __name__ == '__main__':
    app.run(debug=True)




# from flask import Flask, render_template, request
# import pickle
# import numpy as np
# from xgboost import XGBClassifier, Booster


# app = Flask(__name__)

# # Load the pickle model
# filename3 = 'saved_model_xgb.sav'

# load_model=pickle.load(open(filename3, 'rb'))

# # Load the Booster model (JSON format)
# booster = Booster()
# booster.load_model('xgb_model.json')  # Ensure the model has been saved earlier in JSON format


# @app.route('/')
# def home():
#     result = ''
#     return render_template('ermm.html', **locals())

# @app.route('/predict', methods=['POST', 'GET'])
# def predict():
#     x = np.array(request.form['x'], dtype=float)  # Assuming input is numeric
#     processed_data = preprocess_wafer_data(x)  # Preprocess function to prepare input data

#     # Predict using the pickle model
#     result_pickle = load_model.predict(processed_data)[0]

#     # Predict using the Booster model
#     from xgboost import DMatrix
#     processed_dmatrix = DMatrix(processed_data)
#     result_booster = booster.predict(processed_dmatrix)

#     # Example: Combine results or choose one
#     result = f"Pickle Model Result: {result_pickle}, Booster Model Result: {result_booster[0]}"
#     return render_template('ermm.html', **locals())

# if __name__ == '__main__':
#     app.run(debug=True)





# from flask import Flask, render_template, request
# import pickle
# app = Flask(__name__)
# # load the model'


# filename3 = 'savemodle3.sav'
# load_model=pickle.load(open(filename3, 'rb'))

# @app.route('/')
# def home():
#     result = ''
#     return render_template('ermm.html', **locals())

# @app.route('/predict', methods=['POST', 'GET'])
# def predict():
#     x = np.array(request.form['x'])
#     processed_data = preprocess_wafer_data(x)
#     result = load_model.predict(processed_data)[0]
#     return render_template('ermm.html', **locals())


# if __name__ == '__main__':
#     app.run(debug=True)










# # img = x.waferMap[x[i]] //<-col 


# # def cal_den(x):
# #     return 100*(np.sum(x==2)/np.size(x))  

# # def find_regions(x):
# #     rows=np.size(x,axis=0)
# #     cols=np.size(x,axis=1)
# #     ind1=np.arange(0,rows,rows//5)
# #     ind2=np.arange(0,cols,cols//5)
    
# #     reg1=x[ind1[0]:ind1[1],:]
# #     reg3=x[ind1[4]:,:]
# #     reg4=x[:,ind2[0]:ind2[1]]
# #     reg2=x[:,ind2[4]:]

# #     reg5=x[ind1[1]:ind1[2],ind2[1]:ind2[2]]
# #     reg6=x[ind1[1]:ind1[2],ind2[2]:ind2[3]]
# #     reg7=x[ind1[1]:ind1[2],ind2[3]:ind2[4]]
# #     reg8=x[ind1[2]:ind1[3],ind2[1]:ind2[2]]
# #     reg9=x[ind1[2]:ind1[3],ind2[2]:ind2[3]]
# #     reg10=x[ind1[2]:ind1[3],ind2[3]:ind2[4]]
# #     reg11=x[ind1[3]:ind1[4],ind2[1]:ind2[2]]
# #     reg12=x[ind1[3]:ind1[4],ind2[2]:ind2[3]]
# #     reg13=x[ind1[3]:ind1[4],ind2[3]:ind2[4]]
    
# #     fea_reg_den = []
# #     fea_reg_den = [cal_den(reg1),cal_den(reg2),cal_den(reg3),cal_den(reg4),cal_den(reg5),cal_den(reg6),cal_den(reg7),cal_den(reg8),cal_den(reg9),cal_den(reg10),cal_den(reg11),cal_den(reg12),cal_den(reg13)]
# #     return fea_reg_den


# # df_withpattern['fea_reg']=df_withpattern.waferMap.apply(find_regions)

# # def change_val(img):
# #     img[img==1] =0  
# #     return img


# # df_withpattern_copy['new_waferMap'] =df_withpattern_copy.waferMap.apply(change_val)




# # def cubic_inter_mean(img):
# #     theta = np.linspace(0., 180., max(img.shape), endpoint=False)
# #     sinogram = radon(img, theta=theta)
# #     xMean_Row = np.mean(sinogram, axis = 1)
# #     x = np.linspace(1, xMean_Row.size, xMean_Row.size)
# #     y = xMean_Row
# #     f = interpolate.interp1d(x, y, kind = 'cubic')
# #     xnew = np.linspace(1, xMean_Row.size, 20)
# #     ynew = f(xnew)/100   # use interpolation function returned by `interp1d`
# #     return ynew

# # def cubic_inter_std(img):
# #     theta = np.linspace(0., 180., max(img.shape), endpoint=False)
# #     sinogram = radon(img, theta=theta)
# #     xStd_Row = np.std(sinogram, axis=1)
# #     x = np.linspace(1, xStd_Row.size, xStd_Row.size)
# #     y = xStd_Row
# #     f = interpolate.interp1d(x, y, kind = 'cubic')
# #     xnew = np.linspace(1, xStd_Row.size, 20)
# #     ynew = f(xnew)/100   # use interpolation function returned by `interp1d`
# #     return ynew  

# # def cal_dist(img,x,y):
# #     dim0=np.size(img,axis=0)    
# #     dim1=np.size(img,axis=1)
# #     dist = np.sqrt((x-dim0/2)**2+(y-dim1/2)**2)
# #     return dist  


# # def fea_geom(img):
# #     norm_area=img.shape[0]*img.shape[1]
# #     norm_perimeter=np.sqrt((img.shape[0])**2+(img.shape[1])**2)
    
# #     img_labels = measure.label(img, neighbors=4, connectivity=1, background=0)

# #     if img_labels.max()==0:
# #         img_labels[img_labels==0]=1
# #         no_region = 0
# #     else:
# #         info_region = stats.mode(img_labels[img_labels>0], axis = None)
# #         no_region = info_region[0][0]-1       
    
# #     prop = measure.regionprops(img_labels)
# #     prop_area = prop[no_region].area/norm_area
# #     prop_perimeter = prop[no_region].perimeter/norm_perimeter 
    
# #     prop_cent = prop[no_region].local_centroid 
# #     prop_cent = cal_dist(img,prop_cent[0],prop_cent[1])
    
# #     prop_majaxis = prop[no_region].major_axis_length/norm_perimeter 
# #     prop_minaxis = prop[no_region].minor_axis_length/norm_perimeter  
# #     prop_ecc = prop[no_region].eccentricity  
# #     prop_solidity = prop[no_region].solidity  
    
# #     return prop_area,prop_perimeter,prop_majaxis,prop_minaxis,prop_ecc,prop_solidity

# # df_withpattern_copy['fea_geom'] =df_withpattern_copy.waferMap.apply(fea_geom)




# # df_all=df_withpattern_copy.copy()
# # a=[df_all.fea_reg[i] for i in range(df_all.shape[0])] #13
# # b=[df_all.fea_cub_mean[i] for i in range(df_all.shape[0])] #20
# # c=[df_all.fea_cub_std[i] for i in range(df_all.shape[0])] #20
# # d=[df_all.fea_geom[i] for i in range(df_all.shape[0])] #6
# # fea_all = np.concatenate((np.array(a),np.array(b),np.array(c),np.array(d)),axis=1) #59 in total


# # label=[df_all.failureNum[i] for i in range(df_all.shape[0])]
# # label=np.array(label)




# # X = fea_all
# y = label





 

# # for i in range(8):
# #     img = df_withpattern_copy.waferMap[x[i]]




