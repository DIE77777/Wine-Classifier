from pycaret.regression import load_model, predict_model
import streamlit as st
import pandas as pd
import numpy as np
import base64
import io
import requests


def concat(*args):
            strs = [str(arg) for arg in args if not pd.isnull(arg)]
            return ''.join(strs) if strs else np.nan
        
np_concat = np.vectorize(concat)


model = load_model('modelifoods')


#### Funciones para predecir y descargar ####

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


### creo el cuerpo de la app #####  

def run():  

    from PIL import Image
    image = Image.open('iFood.png')
    st.set_page_config(page_title='ifood',page_icon=image)
    image_host = Image.open('imagen.png')
    st.sidebar.info('This app is created to predict campains Response')
    st.sidebar.success('https://www.ifood.com.co/')
    
    st.sidebar.image(image_host)
    
    st.image(image,use_column_width=True,width=100,)
    
    password = st.text_input("Password 🔑 :", value="", type="password")
    if password == '123':
        add_selectbox = st.sidebar.selectbox(
        "How would you like to predict 🔮 ?",
        ("Campains Respose","Nothing"))
    else:
        st.error('passwword incorrecto')
        return

   # st.title("Salario referencia App")
   
    
    if add_selectbox == 'Campains Respose':
        
        st.title("Campains Respose")
        st.warning(' para utilizar esta app por favor cargar en el Browse file el documento enviado para la prueba .csv')
        file_upload = st.file_uploader("Upload csv file for predictions 📁", type=["csv"])

        if st.checkbox("Show Summary of Dataset"):
            data = pd.read_csv(file_upload)
            st.write(data.describe())
                       
                    
        if file_upload is not None:
            data = pd.read_csv(file_upload)
            data['Edad'] =  2020- data['Year_Birth']
            data.drop(['Year_Birth'],axis=1,inplace=True)
            data['Dt_Customer']  = pd.to_datetime(data['Dt_Customer'], format='%Y-%m-%d')
            data['Dt_Customer'] = round(pd.to_numeric((pd.to_datetime('2021-05-01') - data['Dt_Customer']) / np.timedelta64(1, 'M')))
            predictions = predict_model(estimator=model,data=data)
            predictions['Label']=np.where(predictions['Label']==1,'Acepta','Rechaza')                       
            st.write(predictions)
            st.write(get_table_download_link(predictions,'profiles.csv', 'Download predictions  👨‍💻'), unsafe_allow_html=True)
        
    
if __name__ == '__main__':
    run()
