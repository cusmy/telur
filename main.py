import numpy as np
import pandas as pd
import cv2 as cv
import os
import pickle
import warnings
import streamlit as st
from PIL import Image
from skimage.measure import label, regionprops, regionprops_table
from rembg import remove
from skimage.exposure import rescale_intensity
from skimage.morphology import reconstruction
from skimage.feature import graycomatrix, graycoprops


with open('neural network bersih kotor', 'rb') as file:
    mlp = pickle.load(file)

with open('neural network besar kecil', 'rb') as file:
    mlpbk = pickle.load(file)
with open('classifier-bk', 'rb') as file:
    features_scaler_bk = pickle.load(file)
with open('classifier-mlp', 'rb') as file:
    features_scaler_mlp = pickle.load(file)
def run_data(image):
    img = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    img = remove(img)

    img_convert = cv.cvtColor(img, cv.COLOR_BGRA2RGB)
    vectorized = img_convert.reshape((-1,3))
    vectorized = np.float32(vectorized)
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = 2
    attempts=10
    ret,label,center=cv.kmeans(vectorized,K,None,criteria,attempts,cv.KMEANS_PP_CENTERS)
    center = np.uint8(center)
    res = center[label.flatten()]
    result_image2 = res.reshape((img_convert.shape))
    result_image2 = cv.cvtColor(result_image2,cv.COLOR_RGB2GRAY)
    propss = regionprops_table(result_image2, properties=('area_bbox',
                                                        'perimeter',
                                                    'area',
                                                    'extent',
                                                    'eccentricity'
                                                    ))

    df = pd.DataFrame(propss)
    data = df.iloc[[0]]
    X_BK = features_scaler_bk.transform(data[['area_bbox', 'perimeter','area']])
    data
    # X_BK = data[['area_bbox', 'perimeter','area','extent','eccentricity']]
    
    y_hat_bk = mlpbk.predict(X_BK)
    if y_hat_bk == 'besar' :
         st.write("Telor Besar")
    else :
         st.write("Telor kecil")
    # K = 15
    # attempts=10
    # ret,label,center=cv.kmeans(vectorized,K,None,criteria,attempts,cv.KMEANS_PP_CENTERS)
    # center = np.uint8(center)
    # res = center[label.flatten()]
    # result_image3 = res.reshape((img_convert.shape))
    # result_image3 = cv.cvtColor(result_image3,cv.COLOR_RGB2GRAY)
    graycom = graycomatrix(cv.cvtColor(img,cv.COLOR_RGB2GRAY), distances=[5], angles=[0], levels=256,
                        symmetric=True, normed=True)
    contrast = np.array(graycoprops(graycom, 'contrast'))
    dissimilarity = np.array(graycoprops(graycom, 'dissimilarity'))
    homogeneity = np.array(graycoprops(graycom, 'homogeneity'))
    energy = np.array(graycoprops(graycom, 'energy'))
    correlation = np.array(graycoprops(graycom, 'correlation'))
    ASM = np.array(graycoprops(graycom, 'ASM'))
    d = {'contrast': contrast[0], 'dissimilarity': dissimilarity[0], 'homogeneity' : homogeneity[0],'energy':energy[0],'correlation' : correlation[0],'ASM' : ASM[0]}
    df = pd.DataFrame(data=d)
    df
    # X = df[['contrast', 'dissimilarity','homogeneity','energy','correlation','ASM']]
    X = features_scaler_mlp.transform(df[['contrast', 'dissimilarity','homogeneity']])
    yhat = mlp.predict(X)
    if yhat == 'bersih' :
         st.write("Telor Bersih")
    else :
         st.write("Telor Kotor")
      
    col11, col22, col33 = st.columns(3)
    with col11:
        st.image(
        image,
        caption=f"Citra Asli",
        use_column_width=True,
    )

    with col22:
        st.image(
        result_image2,
        caption=f"Kmeans K=2",
        use_column_width=True,
    )

    with col33:
        st.write()
    


file = st.file_uploader("upload gambar Saja", type=["png", "jpg", "jpeg"])
image = []
if file is not None:
    image = Image.open(file)
    
    st.image(
        image,
        caption=f"Berhasil upload gambar",
        use_column_width=True,
    )
img_array = np.array(image,)
btn = st.button("Proses")
if btn:
	run_data(img_array)
else:
	pass