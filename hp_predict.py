import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

st.set_page_config(layout="wide")
st.set_option('deprecation.showPyplotGlobalUse', False)

st.write("""
# House Price Prediction using PAS
This application will try to do prediction of housing price based on its certain feature variables.\n
Dataset used taken from: https://www.kaggle.com/shree1992/housedata
""")
st.write('---')
# Load dataset-----------------------------------------------------------------------------------------
data = pd.read_csv("./data/hse_predx.csv")
#data = data.drop('id', axis=1)
st.header("Dataset Sample")
st.write(data.head())
# View dataset summary---------------------------------------------------------------------------------
st.header("Dataset Summary")
st.write(data.shape)
st.write(data.describe())
st.write('---')
# Specify variables feature and target------------------------------------------------------------------
X = data[['bedrooms','bathrooms','sqft_living','sqft_lot','floors','condition','h_age']].values
Y = data[['price']].values
# Slide for specify Input Parameters --------------------------------------------------------------------
st.sidebar.header('Specify Input: ')
def user_input_features():
    bedrooms    = st.sidebar.slider('No. of bedrooms', int(data['bedrooms'].min()), int(data['bedrooms'].max()))
    bathrooms   = st.sidebar.slider('No. of bathrooms', int(data['bathrooms'].min()), int(data['bathrooms'].max()))
    sqft_living = st.sidebar.slider('Size (House) in Sqft', int(data['sqft_living'].min()), int(data['sqft_living'].max()))
    sqft_lot    = st.sidebar.slider('Size (Area) in Sqft', int(data['sqft_lot'].min()), int(data['sqft_lot'].max()))
    floors      = st.sidebar.slider('No. of floors', int(data['floors'].min()), int(data['floors'].max()))
    condition   = st.sidebar.slider('Condition (1-Poor, 2-Fair, 3-Avg, 4-Good, 5-Superb)', int(data['condition'].min()), int(data['condition'].max()))
    h_age       = st.sidebar.slider('Property Age (deduct from year 2021)', int(data['h_age'].min()), int(data['h_age'].max()))

    datax = {'bedrooms': bedrooms,'bathrooms': bathrooms,'sqft_living': sqft_living, 'sqft_lot': sqft_lot,
            'floors': floors,'condition': condition,'h_age': h_age}
    features = pd.DataFrame(datax, index=[0])
    return features
df = user_input_features()
# Show selected specified input parameters--------------------------------------------------------------------
st.header('Selected Input parameters')
st.write(df)
# Build Regression Model -------------------------------------------------------------------------------------

model = LinearRegression()
model.fit(X, Y)
# Prediction -------------------------------------------------------------------------------------------------
prediction = int(model.predict(df))
st.header('House price prediction: ')
st.write("Predicted house price based on selected parameter is: ", prediction)
st.write("Model accuracy score :", round(model.score(X, Y),4))
st.write('---')
