"""
STREAMLIT APP - Obesity Risk Prediction
Health Awareness Application
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import sqlite3
from datetime import datetime

# Page Configuration
st.set_page_config(
    page_title="HealthAware - Obesity Risk Prediction",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #667eea;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# DATABASE FUNCTIONS
# ============================================================================

def init_db():
    """Initialize SQLite database"""
    conn = sqlite3.connect('predictions.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS predictions
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  timestamp TEXT,
                  predicted_class TEXT,
                  confidence REAL,
                  age INTEGER,
                  gender TEXT,
                  height REAL,
                  weight REAL,
                  bmi REAL,
                  risk_level TEXT)''')
    conn.commit()
    conn.close()

def save_prediction(user_data, predicted_class, confidence, risk_level):
    """Save prediction to database"""
    try:
        conn = sqlite3.connect('predictions.db')
        c = conn.cursor()
        bmi = user_data['Weight'] / (user_data['Height'] ** 2)
        c.execute('''INSERT INTO predictions 
                     (timestamp, predicted_class, confidence, age, gender, height, weight, bmi, risk_level)
                     VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                  (datetime.now().isoformat(), predicted_class, confidence,
                   user_data['Age'], user_data['Gender'], user_data['Height'],
                   user_data['Weight'], bmi, risk_level))
        conn.commit()
        conn.close()
    except Exception as e:
        st.error(f"Error saving prediction: {e}")

# ============================================================================
# MODEL FUNCTIONS
# ============================================================================

@st.cache_resource
def load_model_and_preprocessor():
    """Load trained model and preprocessor"""
    try:
        with open('models/best_model.pkl', 'rb') as f:
            model = pickle.load(f)
        
        with open('models/preprocessor.pkl', 'rb') as f:
            preprocessor_data = pickle.load(f)
            preprocessor = {
                'scaler': preprocessor_data['scaler'],
                'label_encoders': preprocessor_data['label_encoders'],
                'target_encoder': preprocessor_data['target_encoder'],
                'feature_names': preprocessor_data['feature_names']
            }
            class_names = preprocessor_data['target_encoder'].classes_
        
        return model, preprocessor, class_names
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None, None

def preprocess_input(data, preprocessor):
    """Preprocess input data"""
    df = pd.DataFrame([data])
    
    # Feature Engineering
    df['BMI'] = df['Weight'] / (df['Height'] ** 2)
    
    age_val = df['Age'].values[0]
    if age_val <= 18:
        age_group = 'Teen'
    elif age_val <= 30:
        age_group = 'Young_Adult'
    elif age_val <= 45:
        age_group = 'Adult'
    elif age_val <= 60:
        age_group = 'Middle_Age'
    else:
        age_group = 'Senior'
    df['AgeGroup'] = age_group
    
    faf_val = df['FAF'].values[0]
    if faf_val <= 1:
        activity = 'Sedentary'
    elif faf_val <= 3:
        activity = 'Light'
    elif faf_val <= 5:
        activity = 'Moderate'
    else:
        activity = 'Active'
    df['ActivityLevel'] = activity
    
    df['HealthyEatingScore'] = df['FCVC'] + df['NCP'] + (df['CH2O'] / 3)
    df['HighTechUse'] = (df['TUE'] > 4).astype(int)
    df['ActiveTransport'] = df['MTRANS'].isin(['Walking', 'Bike']).astype(int)
    
    favc_val = 1 if str(df['FAVC'].values[0]).lower() in ['yes', '1'] else 0
    df['CalorieVegetableRatio'] = favc_val / (df['FCVC'] + 0.1)
    
    ch2o_val = df['CH2O'].values[0]
    if ch2o_val <= 1.5:
        water = 'Low'
    elif ch2o_val <= 3:
        water = 'Medium'
    else:
        water = 'High'
    df['WaterIntake'] = water
    
    calc_val = 0 if str(df['CALC'].values[0]).lower() == 'no' else 1
    smoke_val = 1 if str(df['SMOKE'].values[0]).lower() in ['yes', '1'] else 0
    df['UnhealthyHabits'] = calc_val + smoke_val
    
    # Encode categoricals
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if col in preprocessor['label_encoders']:
            le = preprocessor['label_encoders'][col]
            val = str(df[col].values[0])
            if val not in le.classes_:
                val = le.classes_[0]
            df[col] = le.transform([val])
    
    # Scale numericals
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    df[numerical_cols] = preprocessor['scaler'].transform(df[numerical_cols])
    
    # Ensure features
    for feature in preprocessor['feature_names']:
        if feature not in df.columns:
            df[feature] = 0
    
    df = df[preprocessor['feature_names']]
    return df

def get_risk_info(predicted_class):
    """Get risk level information"""
    risk_map = {
        'Insufficient_Weight': {'status': 'Underweight', 'color': '#3498db', 'icon': '⚠️', 'risk': 'Moderate'},
        'Normal_Weight': {'status': 'Normal Weight', 'color': '#2ecc71', 'icon': '✅', 'risk': 'Low'},
        'Overweight_Level_I': {'status': 'Overweight (Level I)', 'color': '#f39c12', 'icon': '⚠️', 'risk': 'Moderate'},
        'Overweight_Level_II': {'status': 'Overweight (Level II)', 'color': '#e67e22', 'icon': '⚠️', 'risk': 'High'},
        'Obesity_Type_I': {'status': 'Obesity (Type I)', 'color': '#e74c3c', 'icon': '🚨', 'risk': 'High'},
        'Obesity_Type_II': {'status': 'Obesity (Type II)', 'color': '#c0392b', 'icon': '🚨', 'risk': 'Very High'},
        'Obesity_Type_III': {'status': 'Obesity (Type III)', 'color': '#8b0000', 'icon': '🚨', 'risk': 'Critical'}
    }
    return risk_map.get(predicted_class, risk_map['Normal_Weight'])

# ============================================================================
# MAIN APP
# ============================================================================

def main():
    # Header
    st.markdown('<h1 class="main-header">🏥 HealthAware</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">Obesity Risk Prediction System</p>', unsafe_allow_html=True)
    
    # Initialize
    init_db()
    model, preprocessor, class_names = load_model_and_preprocessor()
    
    if model is None:
        st.error("Failed to load model. Please check that model files exist in the 'models' directory.")
        return
    
    # Sidebar
    st.sidebar.title("📋 Input Data")
    st.sidebar.markdown("---")
    
    # Personal Information
    st.sidebar.subheader("👤 Personal Info")
    gender = st.sidebar.selectbox("Gender", ["Female", "Male"])
    age = st.sidebar.number_input("Age", min_value=10, max_value=100, value=25)
    height = st.sidebar.number_input("Height (meters)", min_value=1.0, max_value=2.5, value=1.7, step=0.01)
    weight = st.sidebar.number_input("Weight (kg)", min_value=30.0, max_value=200.0, value=70.0, step=0.5)
    
    # Calculate BMI
    bmi = weight / (height ** 2)
    st.sidebar.metric("Your BMI", f"{bmi:.2f}")
    
    st.sidebar.markdown("---")
    
    # Family History
    st.sidebar.subheader("👨‍👩‍👧‍👦 Family History")
    family_history = st.sidebar.radio("Family history with overweight?", ["yes", "no"])
    
    st.sidebar.markdown("---")
    
    # Eating Habits
    st.sidebar.subheader("🍽️ Eating Habits")
    favc = st.sidebar.radio("Frequent consumption of high caloric food?", ["yes", "no"])
    fcvc = st.sidebar.slider("Frequency of vegetables consumption (1-3)", 1.0, 3.0, 2.0, 0.5)
    ncp = st.sidebar.slider("Number of main meals per day", 1.0, 4.0, 3.0, 0.5)
    caec = st.sidebar.selectbox("Food consumption between meals", ["no", "Sometimes", "Frequently", "Always"])
    
    st.sidebar.markdown("---")
    
    # Lifestyle
    st.sidebar.subheader("🏃 Lifestyle")
    smoke = st.sidebar.radio("Do you smoke?", ["no", "yes"])
    ch2o = st.sidebar.slider("Daily water intake (liters)", 1.0, 3.0, 2.0, 0.5)
    scc = st.sidebar.radio("Calories consumption monitoring?", ["no", "yes"])
    faf = st.sidebar.slider("Physical activity frequency (days/week)", 0.0, 7.0, 3.0, 0.5)
    tue = st.sidebar.slider("Time using technology devices (hours/day)", 0.0, 12.0, 4.0, 0.5)
    calc = st.sidebar.selectbox("Alcohol consumption", ["no", "Sometimes", "Frequently", "Always"])
    mtrans = st.sidebar.selectbox("Transportation used", ["Walking", "Bike", "Motorbike", "Automobile", "Public_Transportation"])
    
    st.sidebar.markdown("---")
    
    # Predict Button
    if st.sidebar.button("🔮 Predict Risk", use_container_width=True):
        # Prepare data
        user_data = {
            'Gender': gender,
            'Age': age,
            'Height': height,
            'Weight': weight,
            'family_history_with_overweight': family_history,
            'FAVC': favc,
            'FCVC': fcvc,
            'NCP': ncp,
            'CAEC': caec,
            'SMOKE': smoke,
            'CH2O': ch2o,
            'SCC': scc,
            'FAF': faf,
            'TUE': tue,
            'CALC': calc,
            'MTRANS': mtrans
        }
        
        try:
            # Preprocess and predict
            processed_data = preprocess_input(user_data, preprocessor)
            prediction = model.predict(processed_data)[0]
            probabilities = model.predict_proba(processed_data)[0]
            
            predicted_class = class_names[prediction]
            confidence = float(probabilities[prediction])
            
            risk_info = get_risk_info(predicted_class)
            
            # Save to database
            save_prediction(user_data, predicted_class, confidence, risk_info['risk'])
            
            # Display Results
            st.markdown("## 📊 Prediction Results")
            
            # Main result card
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Classification", risk_info['status'])
            
            with col2:
                st.metric("Confidence", f"{confidence*100:.1f}%")
            
            with col3:
                st.metric("Risk Level", risk_info['risk'])
            
            # Risk indicator
            st.markdown(f"""
            <div style="background-color: {risk_info['color']}; color: white; padding: 2rem; 
                        border-radius: 10px; text-align: center; margin: 2rem 0;">
                <h2 style="margin: 0;">{risk_info['icon']} {risk_info['status']}</h2>
                <p style="margin: 0.5rem 0 0 0; font-size: 1.2rem;">Risk Level: {risk_info['risk']}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Probability Distribution
            st.markdown("### 📈 Probability Distribution")
            
            prob_df = pd.DataFrame({
                'Class': [cls.replace('_', ' ') for cls in class_names],
                'Probability': [float(prob)*100 for prob in probabilities]
            }).sort_values('Probability', ascending=False)
            
            st.bar_chart(prob_df.set_index('Class'))
            
            # Top 5 Predictions Table
            st.markdown("### 🔝 Top 5 Predictions")
            top5_df = prob_df.head(5).copy()
            top5_df['Probability'] = top5_df['Probability'].apply(lambda x: f"{x:.2f}%")
            st.table(top5_df)
            
            # Recommendations
            st.markdown("### 💡 Health Recommendations")
            
            if risk_info['risk'] == 'Low':
                st.success("✅ You're in a healthy range! Keep up the good work.")
                st.markdown("""
                - Maintain balanced diet
                - Continue regular exercise
                - Stay hydrated
                - Get regular check-ups
                """)
            elif risk_info['risk'] == 'Moderate':
                st.warning("⚠️ Consider making some lifestyle improvements.")
                st.markdown("""
                - Increase physical activity
                - Monitor calorie intake
                - Eat more vegetables
                - Reduce processed foods
                """)
            else:
                st.error("🚨 Important: Consider consulting a healthcare professional.")
                st.markdown("""
                - Schedule medical consultation
                - Follow structured diet plan
                - Regular exercise program
                - Monitor health metrics
                """)
            
            # Personalized Tips
            st.markdown("### 🎯 Personalized Tips")
            
            if faf < 3:
                st.info(f"🏃 Physical Activity: You exercise {faf} days/week. Try to increase to 3-5 days for better health.")
            
            if ch2o < 2:
                st.info(f"💧 Hydration: You drink {ch2o}L daily. Try to increase to 2-3L for better hydration.")
            
            if fcvc < 2:
                st.info(f"🥗 Nutrition: Vegetable intake is {fcvc}/3. Aim for 5+ servings daily.")
            
            if tue > 6:
                st.info(f"📱 Screen Time: {tue} hours daily is high. Consider reducing to improve health.")
            
            st.success("✅ Prediction complete! Data saved to database.")
            
        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")
    
    # Information Section
    with st.expander("ℹ️ About This App"):
        st.markdown("""
        ### HealthAware - Obesity Risk Prediction System
        
        This application uses machine learning to predict obesity risk levels based on lifestyle and health factors.
        
        **Features:**
        - Real-time risk assessment
        - Personalized health recommendations
        - Data tracking and history
        - Evidence-based predictions
        
        **Disclaimer:** This tool is for informational purposes only and should not replace professional medical advice.
        Always consult with a healthcare provider for medical concerns.
        """)

if __name__ == "__main__":
    main()
