import streamlit as st
import pickle
import numpy as numpy

def load_model():
    with open('saved_steps_prj.pkl', 'rb') as file:
        data = pickle.load(file)  
    return data

data = load_model()

regressor = data['model']
le_manufacturer = data['le_manufacturer']
le_cylinders = data['le_cylinders']
le_drive = data['le_drive']
le_type = data['le_type']
le_paint_colors = data['le_paint_colors']

def show_predict_page():
    st.title("Prediction for used car price")

    #You can write in markdown with """
    st.write("""### We need some information to predict the price""") 

    #Select boxes for manufacturer, cylinders, drive, type, paint colors

    manufacturer = ('acura', 'alfa-romeo', 'aston-martin', 'audi', 'bmw', 'buick',
       'cadillac', 'chevrolet', 'chrysler', 'datsun', 'dodge', 'ferrari',
       'fiat', 'ford', 'gmc', 'harley-davidson', 'honda', 'hyundai',
       'infiniti', 'jaguar', 'jeep', 'kia', 'land rover', 'lexus',
       'lincoln', 'mazda', 'mercedes-benz', 'mercury', 'mini',
       'mitsubishi', 'nissan', 'pontiac', 'porsche', 'ram', 'rover',
       'saturn', 'subaru', 'tesla', 'toyota', 'volkswagen', 'volvo')

    cylinders = ('3 cylinders', '4 cylinders',
       '5 cylinders', '6 cylinders', '8 cylinders', '10 cylinders', '12 cylinders'
    )

    drive = ('4wd', 'fwd', 'rwd')

    type1 = ('bus', 'convertible', 'coupe', 'hatchback', 'mini-van',
       'offroad', 'other', 'pickup', 'sedan', 'truck', 'van', 'wagon')

    paint = ('black', 'blue', 'brown', 'custom', 'green', 'grey', 'orange',
       'purple', 'red', 'silver', 'white', 'yellow')

    #Inside the selection box can be a tuple or a list
    manufacturer = st.selectbox("Manufacturer", manufacturer, )
    cylinders = st.selectbox("Cylinder", cylinders)
    type1 = st.selectbox("Type", type1)
    paint = st.selectbox("Paint", paint)

    drive = st.selectbox('Drive', drive)

    # #This is a button. If you click on a button, this is true (and vice versa)
    ok = st.button('Calculate Price')

    if ok:
        X = numpy.array([[manufacturer, cylinders, drive, type1, paint]])
        X[:, 0] = le_manufacturer.transform(X[:,0])
        X[:, 1] = le_cylinders.transform(X[:,1])
        X[:, 2] = le_drive.transform(X[:,2])
        X[:, 3] = le_type.transform(X[:,3])
        X[:, 4] = le_paint_colors.transform(X[:,4])
        X = X.astype(float)

        price = regressor.predict(X)
        st.subheader(f"The estimated price is ${price[0]:.2f}")




    