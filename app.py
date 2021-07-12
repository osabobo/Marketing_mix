import numpy as np
import pandas as pd
import joblib
import os
import streamlit as st
from xgboost import XGBRegressor
#cv_model = open('model.pkl', 'rb')
cv = joblib.load('model2.pkl')

def prediction(week,tv,radio,newspaper):
    week=week

    tv=float(tv)

    radio=float(radio)
    newspaper=float(newspaper)
     # Making predictions
    value=np.array([[week,tv,radio,newspaper]])

    prediction =round( cv.predict(value)[0],1)

    return prediction

def main ():
    from PIL import Image
    image = Image.open('logo1.jpg')
    image_spam = Image.open('image1.jpg')
    st.image(image,use_column_width=False)

    add_selectbox = st.sidebar.selectbox(
    "How would you like to predict?",
    ("Online", "Batch"))

    st.sidebar.info('This app is created to predict marketing sales advert')


    st.sidebar.image(image_spam)





    st.title("Market sales advert Prediction App")

    if add_selectbox == 'Online':
        week = st.text_input('Weekly')
        tv = st.text_input('Tv')
        radio=st.text_input('Radio')
        newspaper=st.text_input('Newspaper')


        result=""




        if st.button("Predict"):
            result = prediction(week,tv,radio,newspaper)
            st.write("Sales amount")
            st.success(result)








    if add_selectbox == 'Batch':
        st.set_option('deprecation.showfileUploaderEncoding', False)
        file_upload = st.file_uploader("Upload csv file for predictions", type="csv")





        st.title('Make sure the csv File is in the same format  as csv before uploading to avoid Error')

        if file_upload is not None:
            data = pd.read_csv(file_upload)
            data=data.drop('sales', axis=1)



            predictions = cv.predict(data)





            st.write(predictions)



if __name__ == '__main__':
    main()
