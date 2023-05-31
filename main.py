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
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.cluster import KMeans


data = pd.read_csv('out_all.csv',delimiter=',')
X = data[['luas', 'tekstur']]
y = data[['label']]
X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)
k = 4
kmeans = KMeans(n_clusters=k)
kmeans.fit(X_train)
cluster_labels = kmeans.predict(X_train)

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
    attempts=10
    ret,label,center=cv.kmeans(vectorized,2,None,criteria,attempts,cv.KMEANS_PP_CENTERS)
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
    # X_BK = data[['area_bbox', 'perimeter','area','extent','eccentricity']]
    
    y_hat_bk = mlpbk.predict(X_BK)
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
    # X = df[['contrast', 'dissimilarity','homogeneity','energy','correlation','ASM']]
    X = features_scaler_mlp.transform(df[['contrast', 'dissimilarity','homogeneity']])
    yhat = mlp.predict(X)
    # if y_hat_bk == 'besar' and yhat == 'bersih':
    #     predik = "1"
    # elif y_hat_bk == 'kecil' and yhat == 'bersih':
    #     predik = "2"
    # elif  y_hat_bk == 'besar' and yhat == 'kotor':
    #     predik = "3"
    # elif y_hat_bk == 'kecil' and yhat == 'kotor':
    #     predik = "4"
    values = mlpbk.predict_proba(X_BK[:1])
    values = np.max(values)
    vvalue = mlp.predict_proba(X[:1])
    vvalue = np.max(vvalue)
    if y_hat_bk == 'besar':
        luas = 1 * values
    else : 
        luas = 2 * values
    if yhat == 'bersih':
        tekstur = 1 * vvalue
    else :
        tekstur = 2 * vvalue
    # hasil = {'luas': luas, 'tekstur': tekstur}
    # ddata = pd.DataFrame.from_records([{ 'luas': luas, 'tekstur': tekstur}])

    x0_values = np.linspace(0.4, 2.1, 100)
    x1_values = np.linspace(0.4, 2.1, 100)
    coords = np.asarray(np.meshgrid(x0_values, x1_values)).T.reshape(-1,2)
    for i, label in enumerate(cluster_labels):
        plt.text(X_train.iloc[i, 0], X_train.iloc[i, 1], str(label), color='red', fontsize=8, ha='center')
        
    plt.scatter(X_train.iloc[:, 0], X_train.iloc[:, 1], c=cluster_labels, cmap='bwr')
    plt.scatter(luas, tekstur, c='g', marker='x', s=300)
    plt.scatter(coords[:,0], coords[:,1], c=kmeans.predict(coords), cmap='tab10', alpha=0.04)
    plt.xlabel('Luas')
    plt.ylabel('Tekstur')
    plt.title('K-Means Clustering')

    st.pyplot(plt)
    if y_hat_bk == 'besar' and yhat == 'bersih':
        predik = "Lottemart"
    elif y_hat_bk == 'kecil' and yhat == 'bersih':
        predik = "Indomart"
    elif  y_hat_bk == 'besar' and yhat == 'kotor':
        predik = "Pasar"
    elif y_hat_bk == 'kecil' and yhat == 'kotor':
        predik = "Buruk"
    st.write('dari hasil diatas telur tersebut termasuk cluster ***{}*** ditandai dengan mark ***X*** berwarna hijau'.format(predik))
    


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