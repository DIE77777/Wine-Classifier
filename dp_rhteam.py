#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from pycaret.regression import load_model, predict_model
import streamlit as st
import pandas as pd
import numpy as np
import base64

model = load_model('Churn_rh_model')

def predict(model, input_df):
    predictions_df = predict_model(estimator=model, data=input_df)
    predictions = predictions_df['Label'][0]
    return predictions

def get_table_download_link(df, filename, linkname):
    """Generates a link allowing the data in a given panda dataframe to be downloaded
    in:  dataframe
    out: href string
    """
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(
        csv.encode()
    ).decode()  # some strings <-> bytes conversions necessary here
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">{linkname}</a>'
    return href


def run():

    from PIL import Image
    image = Image.open('logo.jpg')
    image_hospital = Image.open('introteam.jpg')

    st.image(image,use_column_width=False)

    add_selectbox = st.sidebar.selectbox(
    "How would you like to predict?",
    ("Online", "Batch"))

    st.sidebar.info('This app is created to predict Churn in Human Resources Team')
    st.sidebar.success('https://alianzateam.com/')
    
    st.sidebar.image(image_hospital)

    st.title("Churn Prediction App")

    if add_selectbox == 'Online':

        Edad = st.number_input('Edad', min_value=1, max_value=1000, value=18)
        Nivel_cargo = st.number_input('Nivel_cargo', min_value=1, max_value=8, value=1)
        ID_Genero = st.selectbox('ID_Genero', ['M', 'F'])              
        Inc_Sal = st.number_input('Inc_Sal', min_value=0.00, max_value=0.10, value=0.01)
        Tipo_cargo = st.selectbox('Tipo cargo', [1,2])
        Salario_Fin = st.number_input('Salario_Fin', min_value=800000, max_value=100000000, value=800000)
        Tipo_nomina = st.selectbox('Tipo_nomina', ['LEY 50 NOMINAL TERMINO INDEFIN','LEY 50 NOMINAL TERMINO FIJO','LEY 50 INTEGRAL'])
        Sidicato = st.selectbox('Sidicato', ['SINDICALIZADO','NO SINDICALIZADO'])
        ID_Estado_civil = st.selectbox('ID_Estado_civil', ['SOL','CAS','ND'])
        Ant_Meses = st.number_input('Ant_Meses', min_value=1, max_value=1000, value=100)
        
        
        output=""

        input_dict = {'Edad' : Edad, 'ID_Genero' : ID_Genero,'ID_Estado_civil':ID_Estado_civil, 'Inc_Sal' : Inc_Sal, 'Tipo_cargo'
                      :Tipo_cargo, 'Salario_Fin' : Salario_Fin,'Tipo_nomina':Tipo_nomina,'Sidicato':Sidicato,'Ant_Meses':Ant_Meses,
                      'Nivel_cargo':Nivel_cargo}
        
        
        input_df = pd.DataFrame([input_dict])

        if st.button("Predict"):
            output = predict(model=model, input_df=input_df)
            #output = '$' + str(output)

        st.success('The output is'+ str(output))

    if add_selectbox == 'Batch':

        file_upload = st.file_uploader("Upload csv file for predictions", type=["xls"])

        if st.checkbox("Show Summary of Dataset"):
            data = pd.read_excel(file_upload)
            st.write(data.describe())
                       
                    
        if file_upload is not None:
            data = pd.read_excel(file_upload)
            predictions = predict_model(estimator=model,data=data)
            predictions['Label']=np.where(predictions['Label']==0,'continua','retiro')
            
            st.write(predictions)
            st.write(get_table_download_link(predictions,'profiles.csv', 'Download predictions'), unsafe_allow_html=True)
        
        
if __name__ == '__main__':
    run()

