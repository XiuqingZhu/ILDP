import streamlit as st
import joblib
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import MACCSkeys
import os
from PIL import Image

# Load model and image
model = joblib.load('./Best_ExtraTreesClassifier_model.joblib')

# Load and display the logo
logo_path = "./logo.png"
logo = Image.open(logo_path)

def get_fingerprints(smiles):
    try:
        # Parse SMILES
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError("Invalid SMILES string")
        
        # Calculate MACCS fingerprints
        maccs_fp = MACCSkeys.GenMACCSKeys(mol)
        maccs_bits = np.array(maccs_fp, dtype=int).tolist()

        return maccs_bits
    except Exception as e:
        st.write("**Invalid SMILES string. Unable to perform subsequent calculations. **")
        return None

def generate_feature_vector(smiles, feature_order):
    maccs_bits = get_fingerprints(smiles)
    if maccs_bits is None:
        return None

    feature_vector = []
    for feature in feature_order:
        if feature.startswith("MACCS_"):
            index = int(feature.split("_")[1]) 
            feature_vector.append(maccs_bits[index])

    return feature_vector

# Streamlit user interface
st.title("Inflammatory Lung Diseases Predictor")

# Create three columns for layout
col1, col2 = st.columns([1, 2])

# Display logo in the first column
with col1:
    st.image(logo)
    st.write("Supported by the service of Xiuqing Zhu at the AI-Drug Lab, the affiliated Brain Hospital, Guangzhou Medical University, China. If you have any questions, please feel free to contact me at 2018760376@gzhmu.edu.cn. ") 

# Define feature names
feature_df = pd.read_csv('./features_for_ML.csv')
feature_names = feature_df['Features'].values.tolist()

# Content in the second column
with col2:
    st.write("**Please enter a SMILE string of a compound to predict its risk of inflammatory lung diseases.**")

    # Smiles: string input
    smiles = st.text_input("SMILE (For example: CCCCOOCC):", value="")

    if st.button("Predict"):
        # Generate feature vector
        feature_vector = generate_feature_vector(smiles, feature_names)

        if feature_vector is None:
            st.write("**Please provide a correct SMILES notation. **")
        else:
            features = np.array([feature_vector])
            
            # Predict class and probabilities
            predicted_class = model.predict(features)[0]
            predicted_proba = model.predict_proba(features)[0]

            # Display prediction results
            st.write(f"**Predicted Class:** {predicted_class}")
            st.write(f"**Prediction Probabilities:** {predicted_proba}")

            # Generate advice based on prediction results
            probability = predicted_proba[predicted_class] * 100

            if predicted_class == 1:
                advice = (
                    f"According to our model, the compound you submitted poses a high risk of interstitial lung disease or pneumonitis. "
                    f"The model predicts that the likelihood of interstitial lung disease or pneumonitis is {probability:.2f}%. "
                    "While this is only an estimation, it indicates that the compound may be at a significant risk. "
                )
            else:
                advice = (
                    f"According to our model, the compound you submitted has a low risk of interstitial lung disease and pneumonitis. "
                    f"The model predicts that the likelihood of not experiencing interstitial lung disease and pneumonitis is {probability:.2f}%. "
                )

            st.write(advice)
