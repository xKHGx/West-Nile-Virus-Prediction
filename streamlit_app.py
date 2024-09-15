# %%
import streamlit as st
import pandas as pd
import joblib
import folium
from streamlit_folium import folium_static
import matplotlib.cm as cm
import matplotlib.colors as colors

# Load the pre-trained model
model = joblib.load('model/XGBClassifier_baseline.pkl')

# Load the weather conditions database
test_df = pd.read_csv('datasets/cleaned_dataset/df_test_modelling.csv')

# Streamlit app title
st.title("West Nile Virus Prediction")

# Add a button to clear the data
if st.button("Clear Data"):
    st.session_state.clear()
    st.experimental_rerun()

# File uploader for CSV
uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file is not None:
    # Read the uploaded CSV file
    input_df = pd.read_csv(uploaded_file)
    
    # Check if required columns are present and standardize column names
    if 'latitude' in input_df.columns and 'longitude' in input_df.columns and 'date' in input_df.columns:
        input_df.rename(columns={'latitude': 'Latitude', 'longitude': 'Longitude', 'date': 'Date'}, inplace=True)
    elif 'Latitude' in input_df.columns and 'Longitude' in input_df.columns and 'Date' in input_df.columns:
        pass
    else:
        st.error("The uploaded CSV does not contain 'Latitude', 'Longitude', and 'Date' columns.")
        st.stop()
    
    # Convert the 'Date' column to numeric format using the specified conversion
    input_df['Date'] = pd.to_datetime(input_df['Date'])
    input_df['Date'] = input_df['Date'].astype('int64') // 10**9 // 86400  # Convert to days since 1970-01-01
    
    # Round the Latitude and Longitude values to ensure consistency
    input_df['Latitude'] = input_df['Latitude'].round(5)
    input_df['Longitude'] = input_df['Longitude'].round(5)
    test_df['Latitude'] = test_df['Latitude'].round(5)
    test_df['Longitude'] = test_df['Longitude'].round(5)
    
    # Perform an inner join based on Latitude, Longitude, and Date
    merged_df = pd.merge(input_df, test_df, on=['Latitude', 'Longitude', 'Date'], how='inner')
    
    if not merged_df.empty:
        # Prepare features for prediction
        features = ['AddressAccuracy', 'AvgSpeed_Station1', 'AvgSpeed_Station2', 'Block',
                    'CodeSum_Recode_Station1', 'CodeSum_Recode_Station2', 'Cool_Station1',
                    'Cool_Station2', 'Date', 'Day', 'DayOfWeek', 'DayOfYear',
                    'DewPoint_Station1', 'DewPoint_Station2', 'Heat_Station1',
                    'Heat_Station2', 'IsWeekend', 'Latitude', 'Longitude', 'Month',
                    'PrecipTotal_Station1', 'PrecipTotal_Station2', 'ResultDir_Station1',
                    'ResultDir_Station2', 'ResultSpeed_Station1', 'ResultSpeed_Station2',
                    'SeaLevel_Station1', 'SeaLevel_Station2', 'Species',
                    'StnPressure_Station1', 'StnPressure_Station2', 'Sunrise_Station1',
                    'Sunset_Station1', 'Tavg_Station1', 'Tavg_Station2', 'Tmax_Station1',
                    'Tmax_Station2', 'Tmin_Station1', 'Tmin_Station2', 'Trap', 'WeekOfYear',
                    'WetBulb_Station1', 'WetBulb_Station2', 'Year', 'zip']

        # Check if all required features are present
        if not all(col in merged_df.columns for col in features):
            st.error(f"The following columns are missing: {set(features) - set(merged_df.columns)}")
            st.stop()
        
        X = merged_df[features]
        
        # Predict Wnv probability
        merged_df['WnvProbability'] = model.predict_proba(X)[:, 1]
        
        # Convert the 'Date' column back to the proper format for display
        merged_df['Date'] = pd.to_datetime(merged_df['Date'] * 86400, unit='s', origin='1970-01-01')
        merged_df['Date'] = merged_df['Date'].dt.strftime('%d/%m/%y')
        
        # Convert probability to percentage
        merged_df['WnvProbability'] = merged_df['WnvProbability'] * 100
        
        # Display results
        st.write("Prediction Results:")
        st.write(merged_df[['Latitude', 'Longitude', 'Date', 'WnvProbability']])
        
        # Create a color map
        norm = colors.Normalize(vmin=0.01, vmax=100)
        cmap = cm.get_cmap('OrRd')
        
        # Create a map
        m = folium.Map(location=[input_df['Latitude'].mean(), input_df['Longitude'].mean()], zoom_start=10)
        
        # Add points to the map with gradient color based on WnvProbability
        for _, row in merged_df.iterrows():
            color = colors.to_hex(cmap(norm(row['WnvProbability'])))
            folium.CircleMarker(
                location=(row['Latitude'], row['Longitude']),
                radius=4,
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.6,
                popup=f"Probability: {row['WnvProbability']:.2f}%"
            ).add_to(m)
        
        # Display the map
        folium_static(m)
    else:
        st.warning("No matching weather conditions found for the provided coordinates.")
else:
    st.info("Please upload a CSV file containing Latitude, Longitude, and Date.")



