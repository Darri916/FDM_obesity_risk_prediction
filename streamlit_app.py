"""
Obesity Risk Prediction System - Streamlit Deployment with AI Chatbot
No Database - Session State Only
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
from datetime import datetime
from io import BytesIO
import plotly.express as px

# PDF Generation
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.units import inch

# Page Configuration
st.set_page_config(
    page_title="Obesity Risk Prediction",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #667eea;
        text-align: center;
        font-weight: bold;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        font-weight: 600;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 20px rgba(102, 126, 234, 0.3);
    }
    
    /* Chat Messages Styling */
    .chat-message {
        margin: 20px 0;
        padding: 15px 20px;
        border-radius: 18px;
        max-width: 85%;
        animation: slideIn 0.4s ease;
        box-shadow: 0 2px 10px rgba(0,0,0,0.08);
        line-height: 1.6;
        font-size: 0.98em;
    }
    
    .chat-message.assistant {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        margin-right: auto;
        border-bottom-left-radius: 5px;
        position: relative;
    }
    
    .chat-message.assistant::after {
        content: '';
        position: absolute;
        bottom: 0;
        left: -8px;
        width: 0;
        height: 0;
        border-style: solid;
        border-width: 0 0 20px 8px;
        border-color: transparent transparent #764ba2 transparent;
    }
    
    .chat-message.user {
        background: #ffffff;
        color: #333;
        margin-left: auto;
        border-bottom-right-radius: 5px;
        text-align: right;
        border: 2px solid #e8e8e8;
        position: relative;
    }
    
    .chat-message.user::after {
        content: '';
        position: absolute;
        bottom: 0;
        right: -8px;
        width: 0;
        height: 0;
        border-style: solid;
        border-width: 0 8px 20px 0;
        border-color: transparent transparent #ffffff transparent;
    }
    
    .chat-message.assistant::before {
        content: "🤖 AI Assistant";
        font-size: 0.75em;
        opacity: 0.9;
        display: block;
        margin-bottom: 5px;
        font-weight: 600;
    }
    
    .chat-message.user::before {
        content: "You 👤";
        font-size: 0.75em;
        opacity: 0.7;
        display: block;
        margin-bottom: 5px;
        font-weight: 600;
    }
    
    @keyframes slideIn {
        from {
            opacity: 0;
            transform: translateY(15px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    /* Progress Bar */
    .progress-bar-container {
        background: #f0f0f0;
        height: 8px;
        border-radius: 10px;
        margin: 15px 0;
        overflow: hidden;
    }
    
    .progress-bar-fill {
        height: 100%;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        transition: width 0.5s ease;
        border-radius: 10px;
    }
    
    /* Recommendation Cards */
    .recommendation-card {
        background: #f8f9fa;
        border-left: 4px solid #667eea;
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
        transition: all 0.3s;
    }
    
    .recommendation-card:hover {
        transform: translateX(5px);
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.15);
    }
    
    .recommendation-category {
        font-weight: 700;
        color: #667eea;
        margin-bottom: 8px;
        font-size: 1.1em;
    }
    
    .recommendation-advice {
        color: #555;
        line-height: 1.6;
    }
    
    /* Health Tips List */
    .health-tip {
        padding: 12px 0;
        border-bottom: 1px solid #eee;
        display: flex;
        align-items: start;
    }
    
    .health-tip:last-child {
        border-bottom: none;
    }
    
    .health-tip::before {
        content: "✓";
        color: #2ecc71;
        font-weight: bold;
        margin-right: 12px;
        font-size: 1.3em;
        flex-shrink: 0;
    }
    
    /* Info Box */
    .info-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 20px;
        border-radius: 12px;
        margin: 15px 0;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.2);
    }
    
    /* Result Cards */
    .result-metric {
        background: white;
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        margin: 10px 0;
    }
    
    .metric-value {
        font-size: 2em;
        font-weight: bold;
        color: #667eea;
        margin-bottom: 5px;
    }
    
    .metric-label {
        color: #666;
        font-size: 0.9em;
    }
    
    /* Chat Container */
    .chat-container {
        max-height: 500px;
        overflow-y: auto;
        background: #f8f9fa;
        padding: 20px;
        border-radius: 15px;
        margin: 15px 0;
    }
    
    /* Quick Reply Buttons */
    .quick-reply {
        display: inline-block;
        padding: 8px 16px;
        margin: 5px;
        background: white;
        color: #667eea;
        border: 2px solid #667eea;
        border-radius: 20px;
        cursor: pointer;
        transition: all 0.3s;
        font-weight: 500;
    }
    
    .quick-reply:hover {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        transform: translateY(-2px);
    }
</style>
""", unsafe_allow_html=True)

# Load Models
@st.cache_resource
def load_models():
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
        st.error(f"Error loading models: {str(e)}")
        st.stop()

model, preprocessor, class_names = load_models()

# Initialize Session State
if 'predictions_history' not in st.session_state:
    st.session_state.predictions_history = []

if 'chat_step' not in st.session_state:
    st.session_state.chat_step = 0
    
if 'chat_data' not in st.session_state:
    st.session_state.chat_data = {}
    
if 'chat_messages' not in st.session_state:
    st.session_state.chat_messages = []

# Conversation Flow for AI Assistant
CONVERSATION_FLOW = [
    {"field": "Gender", "question": "Hi! I'm your AI Health Assistant. What's your gender? (Male/Female)", "type": "choice", "options": ["Male", "Female"]},
    {"field": "Age", "question": "Great! How old are you?", "type": "number", "min": 10, "max": 100},
    {"field": "Height", "question": "What's your height in meters? (e.g., 1.75)", "type": "number", "min": 1.0, "max": 2.5},
    {"field": "Weight", "question": "What's your weight in kilograms?", "type": "number", "min": 30, "max": 300},
    {"field": "family_history_with_overweight", "question": "Does anyone in your family have a history of being overweight? (yes/no)", "type": "choice", "options": ["yes", "no"]},
    {"field": "FAVC", "question": "Do you frequently eat high caloric food? (yes/no)", "type": "choice", "options": ["yes", "no"]},
    {"field": "FCVC", "question": "How often do you eat vegetables? (Rate from 1-3, where 1=rarely, 3=always)", "type": "number", "min": 1, "max": 3},
    {"field": "NCP", "question": "How many main meals do you have per day? (1-4)", "type": "number", "min": 1, "max": 4},
    {"field": "CAEC", "question": "Do you eat food between meals? (no/Sometimes/Frequently/Always)", "type": "choice", "options": ["no", "Sometimes", "Frequently", "Always"]},
    {"field": "SMOKE", "question": "Do you smoke? (yes/no)", "type": "choice", "options": ["yes", "no"]},
    {"field": "CH2O", "question": "How much water do you drink daily in liters? (0.5 to 5)", "type": "number", "min": 0.5, "max": 5},
    {"field": "SCC", "question": "Do you monitor your calorie intake? (yes/no)", "type": "choice", "options": ["yes", "no"]},
    {"field": "FAF", "question": "How many days per week do you do physical activity? (0-7)", "type": "number", "min": 0, "max": 7},
    {"field": "TUE", "question": "How many hours per day do you use technology devices? (0-12)", "type": "number", "min": 0, "max": 12},
    {"field": "CALC", "question": "How often do you drink alcohol? (no/Sometimes/Frequently/Always)", "type": "choice", "options": ["no", "Sometimes", "Frequently", "Always"]},
    {"field": "MTRANS", "question": "What's your primary mode of transportation?", "type": "choice", "options": ["Walking", "Bike", "Public_Transportation", "Automobile", "Motorbike"]}
]

# Preprocessing Function
def preprocess_input(data):
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

# Recommendations Function
def get_recommendations(predicted_class, user_data):
    base_recommendations = {
        'Insufficient_Weight': {
            'status': 'Underweight',
            'color': '#3498db',
            'icon': '⚠️',
            'risk_level': 'Moderate',
            'general_advice': [
                'Increase caloric intake with nutrient-dense foods',
                'Eat more frequently (5-6 smaller meals per day)',
                'Include protein-rich foods in every meal',
                'Consider strength training to build muscle mass',
                'Consult with a nutritionist for personalized meal plans'
            ]
        },
        'Normal_Weight': {
            'status': 'Normal Weight - Healthy Range',
            'color': '#2ecc71',
            'icon': '✅',
            'risk_level': 'Low',
            'general_advice': [
                'Maintain current healthy lifestyle habits',
                'Continue balanced diet with variety of foods',
                'Keep regular physical activity routine',
                'Stay hydrated with adequate water intake',
                'Get regular health check-ups annually'
            ]
        },
        'Overweight_Level_I': {
            'status': 'Overweight (Level I)',
            'color': '#f39c12',
            'icon': '⚠️',
            'risk_level': 'Moderate',
            'general_advice': [
                'Start moderate calorie reduction (300-500 cal/day)',
                'Increase physical activity to 150 minutes per week',
                'Reduce consumption of high-calorie processed foods',
                'Monitor portion sizes carefully',
                'Track your progress weekly'
            ]
        },
        'Overweight_Level_II': {
            'status': 'Overweight (Level II)',
            'color': '#e67e22',
            'icon': '⚠️',
            'risk_level': 'High',
            'general_advice': [
                'Implement structured meal planning and tracking',
                'Increase exercise to 200-300 minutes per week',
                'Eliminate sugary drinks and highly processed foods',
                'Join a weight management program or support group',
                'Consider working with a healthcare provider'
            ]
        },
        'Obesity_Type_I': {
            'status': 'Obesity (Type I)',
            'color': '#e74c3c',
            'icon': '🚨',
            'risk_level': 'High',
            'general_advice': [
                'Seek professional medical evaluation immediately',
                'Follow prescribed diet and exercise plan strictly',
                'Monitor blood pressure and glucose levels regularly',
                'Consider behavioral therapy for lifestyle changes',
                'Schedule regular follow-ups with healthcare team'
            ]
        },
        'Obesity_Type_II': {
            'status': 'Obesity (Type II)',
            'color': '#c0392b',
            'icon': '🚨',
            'risk_level': 'Very High',
            'general_advice': [
                'Urgent medical consultation required',
                'Comprehensive weight management program needed',
                'Regular cardiovascular health monitoring essential',
                'Consider medical interventions under supervision',
                'Psychological support for sustainable lifestyle changes'
            ]
        },
        'Obesity_Type_III': {
            'status': 'Obesity (Type III) - Severe',
            'color': '#8b0000',
            'icon': '🚨',
            'risk_level': 'Critical',
            'general_advice': [
                'Immediate medical attention strongly recommended',
                'Multidisciplinary treatment approach required',
                'Intensive lifestyle intervention program essential',
                'Evaluate for bariatric surgery options',
                'Continuous medical monitoring critical'
            ]
        }
    }
    
    info = base_recommendations.get(predicted_class, base_recommendations['Normal_Weight'])
    personalized = []
    
    faf_val = user_data.get('FAF', 0)
    if faf_val < 2:
        personalized.append({
            'category': 'Physical Activity',
            'advice': f"Your activity level is low ({faf_val} days/week). Aim for 3-5 days/week."
        })
    
    ch2o_val = user_data.get('CH2O', 0)
    if ch2o_val < 2:
        personalized.append({
            'category': 'Hydration',
            'advice': f"You drink {ch2o_val}L daily. Increase to 2-3L for better health."
        })
    
    if user_data.get('FCVC', 0) < 2:
        personalized.append({
            'category': 'Nutrition',
            'advice': f"Vegetable intake needs improvement. Aim for 5+ servings daily."
        })
    
    bmi = user_data.get('Weight', 0) / (user_data.get('Height', 1) ** 2) if user_data.get('Height', 0) > 0 else 0
    if bmi > 0:
        bmi_status = "healthy range" if 18.5 <= bmi < 25 else "underweight" if bmi < 18.5 else "overweight" if 25 <= bmi < 30 else "obesity range"
        personalized.append({
            'category': 'BMI Analysis',
            'advice': f"Your BMI is {bmi:.1f}, which is in the {bmi_status}."
        })
    
    return {
        **info,
        'general_recommendations': info['general_advice'],
        'personalized_recommendations': personalized
    }

# PDF Generation
def generate_pdf_report(prediction_data, user_data):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, topMargin=0.5*inch)
    story = []
    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.HexColor('#667eea'),
        spaceAfter=20,
        alignment=1
    )
    
    story.append(Paragraph("Obesity Risk Health Report", title_style))
    story.append(Spacer(1, 0.2*inch))
    
    date_text = f"<b>Generated:</b> {datetime.now().strftime('%B %d, %Y at %I:%M %p')}"
    story.append(Paragraph(date_text, styles['Normal']))
    story.append(Spacer(1, 0.3*inch))
    
    # User Info
    bmi = user_data.get('Weight', 0) / (user_data.get('Height', 1)**2) if user_data.get('Height', 0) > 0 else 0
    
    user_info_data = [
        ['Gender:', user_data.get('Gender', 'N/A')],
        ['Age:', f"{user_data.get('Age', 'N/A')} years"],
        ['Height:', f"{user_data.get('Height', 'N/A')} meters"],
        ['Weight:', f"{user_data.get('Weight', 'N/A')} kg"],
        ['BMI:', f"{bmi:.2f}"],
    ]
    
    user_table = Table(user_info_data, colWidths=[2*inch, 4*inch])
    user_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#f0f0f0')),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey)
    ]))
    story.append(user_table)
    story.append(Spacer(1, 0.3*inch))
    
    # Results
    prediction = prediction_data['prediction']
    result_data = [
        ['Classification:', prediction['status']],
        ['Risk Level:', prediction['risk_level']],
        ['Confidence:', f"{prediction['confidence']*100:.1f}%"],
    ]
    
    result_table = Table(result_data, colWidths=[2*inch, 4*inch])
    result_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#e8f4f8')),
        ('GRID', (0, 0), (-1, -1), 1, colors.white)
    ]))
    story.append(result_table)
    
    doc.build(story)
    buffer.seek(0)
    return buffer

# Main App
st.markdown('<h1 class="main-header">🏥 Obesity Risk Prediction System</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">AI-Powered Health Assessment Platform</p>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("Navigation")
    page = st.radio("Select Page", ["Single Prediction", "AI Health Assistant", "Batch Upload"])
    
    st.markdown("---")
    st.info("This is an AI assessment tool. Always consult healthcare professionals for medical advice.")

if page == "Single Prediction":
    st.header("Enter Your Health Information")
    
    # Info banner
    st.markdown("""
    <div class='info-box'>
        <h3 style='margin: 0 0 10px 0; font-size: 1.5em;'>🔍 Individual Health Assessment</h3>
        <p style='margin: 0; opacity: 0.95; font-size: 1.05em; line-height: 1.6;'>
            Complete the form below to get a personalized obesity risk assessment with tailored health recommendations.
            All fields are required for accurate prediction.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    with st.form("prediction_form"):
        st.markdown("""
        <div style='background: #f8f9fa; padding: 20px; border-radius: 15px; margin: 20px 0;'>
            <h4 style='color: #667eea; margin-top: 0;'>📝 Personal Information</h4>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            gender = st.selectbox("Gender", ["Male", "Female"])
            age = st.number_input("Age", min_value=10, max_value=100, value=25)
            height = st.number_input("Height (meters)", min_value=1.0, max_value=2.5, value=1.75, step=0.01)
            weight = st.number_input("Weight (kg)", min_value=30.0, max_value=200.0, value=70.0, step=0.1)
            family_history = st.selectbox("Family History of Overweight", ["yes", "no"])
        
        with col2:
            favc = st.selectbox("Frequent High Caloric Food", ["yes", "no"])
            fcvc = st.number_input("Vegetable Consumption (1-3)", min_value=1.0, max_value=3.0, value=2.0, step=0.1)
            ncp = st.number_input("Main Meals Per Day", min_value=1.0, max_value=4.0, value=3.0, step=0.1)
            caec = st.selectbox("Food Between Meals", ["no", "Sometimes", "Frequently", "Always"])
            smoke = st.selectbox("Do You Smoke?", ["no", "yes"])
        
        with col3:
            ch2o = st.number_input("Water Intake (Liters/Day)", min_value=0.5, max_value=5.0, value=2.0, step=0.1)
            scc = st.selectbox("Calorie Monitoring", ["no", "yes"])
            faf = st.number_input("Physical Activity (Days/Week)", min_value=0.0, max_value=7.0, value=3.0, step=0.5)
            tue = st.number_input("Technology Use (Hours/Day)", min_value=0.0, max_value=12.0, value=2.0, step=0.5)
            calc = st.selectbox("Alcohol Consumption", ["no", "Sometimes", "Frequently", "Always"])
            mtrans = st.selectbox("Transportation Mode", ["Walking", "Bike", "Public_Transportation", "Automobile", "Motorbike"])
        
        submitted = st.form_submit_button("Predict My Obesity Risk")
    
    if submitted:
        user_data = {
            'Gender': gender, 'Age': age, 'Height': height, 'Weight': weight,
            'family_history_with_overweight': family_history, 'FAVC': favc,
            'FCVC': fcvc, 'NCP': ncp, 'CAEC': caec, 'SMOKE': smoke,
            'CH2O': ch2o, 'SCC': scc, 'FAF': faf, 'TUE': tue,
            'CALC': calc, 'MTRANS': mtrans
        }
        
        with st.spinner("Analyzing your health data..."):
            processed_data = preprocess_input(user_data)
            prediction = model.predict(processed_data)[0]
            probabilities = model.predict_proba(processed_data)[0]
            
            predicted_class = class_names[prediction]
            confidence = float(probabilities[prediction])
            
            recommendations_data = get_recommendations(predicted_class, user_data)
            
            # Store in session
            st.session_state.predictions_history.append({
                'timestamp': datetime.now(),
                'class': predicted_class,
                'confidence': confidence,
                'bmi': weight / (height ** 2)
            })
            
            # Display Results
            st.success("Analysis Complete!")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Classification", recommendations_data['status'])
            with col2:
                st.metric("Risk Level", recommendations_data['risk_level'])
            with col3:
                st.metric("Confidence", f"{confidence*100:.1f}%")
            
            # Probability Chart
            st.subheader("Probability Distribution")
            prob_df = pd.DataFrame({
                'Class': [c.replace('_', ' ') for c in class_names],
                'Probability': probabilities
            }).sort_values('Probability', ascending=False).head(5)
            
            fig = px.bar(prob_df, x='Probability', y='Class', orientation='h',
                        color='Probability', color_continuous_scale='Blues')
            st.plotly_chart(fig, use_container_width=True)
            
            # Recommendations with enhanced styling
            st.subheader("💡 Your Health Recommendations")
            
            st.markdown("""
            <div style='background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%); 
                        padding: 20px; border-radius: 15px; margin-bottom: 20px;'>
                <h4 style='color: #667eea; margin-bottom: 15px;'>General Health Guidelines</h4>
            </div>
            """, unsafe_allow_html=True)
            
            for idx, rec in enumerate(recommendations_data['general_recommendations'], 1):
                st.markdown(f"""
                <div class='health-tip' style='padding: 12px; margin: 8px 0;'>
                    <span style='color: #2ecc71; font-weight: bold; font-size: 1.3em; margin-right: 12px;'>✓</span>
                    <span style='color: #333; line-height: 1.6;'>{rec}</span>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            st.markdown("""
            <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                        color: white; padding: 20px; border-radius: 15px; margin: 20px 0;'>
                <h4 style='margin-bottom: 15px;'>🎯 Personalized Insights Based on Your Data</h4>
                <p style='opacity: 0.9; margin: 0;'>These recommendations are tailored specifically to your health profile</p>
            </div>
            """, unsafe_allow_html=True)
            
            for rec in recommendations_data['personalized_recommendations']:
                st.markdown(f"""
                <div class='recommendation-card' style='background: #f8f9fa; border-left: 4px solid #667eea; 
                            border-radius: 8px; padding: 15px; margin: 15px 0; transition: all 0.3s;'>
                    <div class='recommendation-category' style='font-weight: 700; color: #667eea; 
                                margin-bottom: 8px; font-size: 1.1em;'>
                        {rec['category']}
                    </div>
                    <div class='recommendation-advice' style='color: #555; line-height: 1.6;'>
                        {rec['advice']}
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            # PDF Download
            prediction_data_for_pdf = {
                'prediction': {
                    'class': predicted_class,
                    'confidence': confidence,
                    'status': recommendations_data['status'],
                    'risk_level': recommendations_data['risk_level'],
                    'color': recommendations_data['color'],
                    'icon': recommendations_data['icon']
                },
                'top_predictions': []
            }
            
            pdf_buffer = generate_pdf_report(prediction_data_for_pdf, user_data)
            st.download_button(
                label="Download PDF Report",
                data=pdf_buffer,
                file_name=f"health_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                mime="application/pdf"
            )

elif page == "AI Health Assistant":
    st.header("AI Health Assistant - Conversational Assessment")
    
    # Welcome banner
    st.markdown("""
    <div class='info-box' style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                color: white; padding: 25px; border-radius: 15px; margin-bottom: 20px; 
                box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);'>
        <h3 style='margin: 0 0 10px 0; font-size: 1.5em;'>👋 Welcome to Your Personal Health Assistant</h3>
        <p style='margin: 0; opacity: 0.95; font-size: 1.05em; line-height: 1.6;'>
            I'll guide you through a friendly conversation to assess your obesity risk. 
            Just answer my questions naturally, and I'll provide personalized insights along the way!
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Progress bar
    if st.session_state.chat_step > 0:
        progress = (st.session_state.chat_step / len(CONVERSATION_FLOW)) * 100
        st.markdown(f"""
        <div class='progress-bar-container'>
            <div class='progress-bar-fill' style='width: {progress}%;'></div>
        </div>
        <p style='text-align: center; color: #666; font-size: 0.9em;'>
            Question {st.session_state.chat_step} of {len(CONVERSATION_FLOW)}
        </p>
        """, unsafe_allow_html=True)
    
    # Chat container
    st.markdown("<div class='chat-container' style='max-height: 500px; overflow-y: auto; background: #f8f9fa; padding: 20px; border-radius: 15px;'>", unsafe_allow_html=True)
    
    # Display chat messages
    for msg in st.session_state.chat_messages:
        if msg['role'] == 'assistant':
            st.markdown(f"""
            <div class='chat-message assistant'>
                {msg["content"]}
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class='chat-message user'>
                {msg["content"]}
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Current question
    if st.session_state.chat_step < len(CONVERSATION_FLOW):
        current_q = CONVERSATION_FLOW[st.session_state.chat_step]
        
        st.markdown(f"""
        <div class='chat-message assistant' style='margin-top: 20px;'>
            {current_q["question"]}
        </div>
        """, unsafe_allow_html=True)
        
        # Input based on type
        if current_q['type'] == 'choice':
            user_input = st.selectbox("Your answer:", current_q['options'], key=f"q_{st.session_state.chat_step}")
        else:
            user_input = st.number_input("Your answer:", 
                                        min_value=float(current_q['min']), 
                                        max_value=float(current_q['max']),
                                        value=float(current_q['min']),
                                        step=0.1 if current_q['min'] < 10 else 1.0,
                                        key=f"q_{st.session_state.chat_step}")
        
        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("📤 Submit Answer", use_container_width=True):
                st.session_state.chat_messages.append({'role': 'assistant', 'content': current_q['question']})
                st.session_state.chat_messages.append({'role': 'user', 'content': str(user_input)})
                st.session_state.chat_data[current_q['field']] = user_input
                st.session_state.chat_step += 1
                st.rerun()
        
        with col2:
            if st.button("🔄 Start Over", use_container_width=True):
                st.session_state.chat_step = 0
                st.session_state.chat_data = {}
                st.session_state.chat_messages = []
                st.rerun()
    
    else:
        # All questions answered - make prediction
        st.success("✨ Assessment Complete! Analyzing your data...")
        
        with st.spinner("Processing your health information..."):
            processed_data = preprocess_input(st.session_state.chat_data)
            prediction = model.predict(processed_data)[0]
            probabilities = model.predict_proba(processed_data)[0]
            
            predicted_class = class_names[prediction]
            confidence = float(probabilities[prediction])
            
            recommendations_data = get_recommendations(predicted_class, st.session_state.chat_data)
            
            # Display Results
            st.markdown("<br>", unsafe_allow_html=True)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown(f"""
                <div class='result-metric'>
                    <div class='metric-value'>{recommendations_data['icon']}</div>
                    <div class='metric-label' style='font-size: 1.1em; font-weight: 600;'>
                        {recommendations_data['status']}
                    </div>
                </div>
                """, unsafe_allow_html=True)
            with col2:
                st.markdown(f"""
                <div class='result-metric'>
                    <div class='metric-value' style='color: {recommendations_data["color"]};'>
                        {recommendations_data['risk_level']}
                    </div>
                    <div class='metric-label'>Risk Level</div>
                </div>
                """, unsafe_allow_html=True)
            with col3:
                st.markdown(f"""
                <div class='result-metric'>
                    <div class='metric-value'>{confidence*100:.1f}%</div>
                    <div class='metric-label'>Confidence</div>
                </div>
                """, unsafe_allow_html=True)
            
            # Recommendations
            st.markdown("<br>", unsafe_allow_html=True)
            
            st.markdown("""
            <div style='background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%); 
                        padding: 20px; border-radius: 15px; margin-bottom: 20px;'>
                <h4 style='color: #667eea; margin-bottom: 15px;'>💡 Your Health Recommendations</h4>
            </div>
            """, unsafe_allow_html=True)
            
            for rec in recommendations_data['general_recommendations']:
                st.markdown(f"""
                <div class='health-tip' style='padding: 12px; margin: 8px 0;'>
                    <span style='color: #2ecc71; font-weight: bold; font-size: 1.3em; margin-right: 12px;'>✓</span>
                    <span style='color: #333; line-height: 1.6;'>{rec}</span>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            st.markdown("""
            <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                        color: white; padding: 20px; border-radius: 15px; margin: 20px 0;'>
                <h4 style='margin-bottom: 15px;'>🎯 Personalized Insights Based on Your Data</h4>
                <p style='opacity: 0.9; margin: 0;'>These recommendations are tailored specifically to your health profile</p>
            </div>
            """, unsafe_allow_html=True)
            
            for rec in recommendations_data['personalized_recommendations']:
                st.markdown(f"""
                <div class='recommendation-card'>
                    <div class='recommendation-category'>{rec['category']}</div>
                    <div class='recommendation-advice'>{rec['advice']}</div>
                </div>
                """, unsafe_allow_html=True)
            
            # PDF Download
            prediction_data_for_pdf = {
                'prediction': {
                    'class': predicted_class,
                    'confidence': confidence,
                    'status': recommendations_data['status'],
                    'risk_level': recommendations_data['risk_level'],
                    'color': recommendations_data['color'],
                    'icon': recommendations_data['icon']
                },
                'top_predictions': []
            }
            
            pdf_buffer = generate_pdf_report(prediction_data_for_pdf, st.session_state.chat_data)
            st.download_button(
                label="📄 Download PDF Report",
                data=pdf_buffer,
                file_name=f"health_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                mime="application/pdf"
            )
            
            if st.button("🔄 Start New Assessment", use_container_width=True):
                st.session_state.chat_step = 0
                st.session_state.chat_data = {}
                st.session_state.chat_messages = []
                st.rerun()

elif page == "Batch Upload":
    st.header("Batch Predictions from CSV")
    
    # Welcome info box
    st.markdown("""
    <div class='info-box'>
        <h3 style='margin: 0 0 10px 0; font-size: 1.5em;'>📊 Bulk Health Assessment</h3>
        <p style='margin: 0; opacity: 0.95; font-size: 1.05em; line-height: 1.6;'>
            Upload a CSV file to get obesity risk predictions for multiple individuals at once. 
            Perfect for health screenings, research studies, or organizational wellness programs.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Instructions section
    st.markdown("""
    <div style='background: #f8f9fa; padding: 20px; border-radius: 15px; margin: 20px 0; border-left: 4px solid #667eea;'>
        <h4 style='color: #667eea; margin-top: 0;'>📋 How to Use Batch Upload</h4>
        <ol style='margin: 10px 0; padding-left: 20px; line-height: 2;'>
            <li><strong>Download</strong> the CSV template below</li>
            <li><strong>Fill in</strong> your data (one person per row)</li>
            <li><strong>Save</strong> the file in CSV format</li>
            <li><strong>Upload</strong> using the file uploader</li>
            <li><strong>Process</strong> and download results with all data</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)
    
    # Required columns section
    st.markdown("""
    <div style='background: linear-gradient(135deg, #fff3cd 0%, #ffe69c 100%); 
                padding: 20px; border-radius: 15px; margin: 20px 0; 
                border-left: 4px solid #ffc107; box-shadow: 0 2px 10px rgba(255, 193, 7, 0.1);'>
        <h4 style='color: #856404; margin-top: 0; display: flex; align-items: center;'>
            ⚠️ Required Columns
        </h4>
        <div style='background: white; padding: 15px; border-radius: 10px; margin-top: 10px;'>
            <div style='display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 10px; font-size: 0.9em;'>
                <div><strong>• Gender</strong> (Male/Female)</div>
                <div><strong>• Age</strong> (10-100)</div>
                <div><strong>• Height</strong> (1.0-2.5 meters)</div>
                <div><strong>• Weight</strong> (30-300 kg)</div>
                <div><strong>• family_history_with_overweight</strong> (yes/no)</div>
                <div><strong>• FAVC</strong> (yes/no)</div>
                <div><strong>• FCVC</strong> (1-3)</div>
                <div><strong>• NCP</strong> (1-4)</div>
                <div><strong>• CAEC</strong> (no/Sometimes/Frequently/Always)</div>
                <div><strong>• SMOKE</strong> (yes/no)</div>
                <div><strong>• CH2O</strong> (0.5-5 liters)</div>
                <div><strong>• SCC</strong> (yes/no)</div>
                <div><strong>• FAF</strong> (0-7 days)</div>
                <div><strong>• TUE</strong> (0-12 hours)</div>
                <div><strong>• CALC</strong> (no/Sometimes/Frequently/Always)</div>
                <div><strong>• MTRANS</strong> (Walking/Bike/Public_Transportation/Automobile/Motorbike)</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Template download with styling
    col1, col2 = st.columns([1, 2])
    with col1:
        template_data = {
            'Gender': ['Male', 'Female'],
            'Age': [23, 28],
            'Height': [1.75, 1.62],
            'Weight': [89, 65],
            'family_history_with_overweight': ['no', 'yes'],
            'FAVC': ['yes', 'no'],
            'FCVC': [3, 2.5],
            'NCP': [4, 3],
            'CAEC': ['Sometimes', 'Sometimes'],
            'SMOKE': ['no', 'no'],
            'CH2O': [3, 2.5],
            'SCC': ['no', 'yes'],
            'FAF': [3, 4],
            'TUE': [2, 1],
            'CALC': ['no', 'Sometimes'],
            'MTRANS': ['Automobile', 'Walking']
        }
        
        template_df = pd.DataFrame(template_data)
        csv = template_df.to_csv(index=False)
        
        st.download_button(
            label="📥 Download CSV Template",
            data=csv,
            file_name="obesity_prediction_template.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    with col2:
        st.markdown("""
        <div style='padding: 10px; background: #e8f4f8; border-radius: 10px; height: 100%; display: flex; align-items: center;'>
            <p style='margin: 0; color: #0c5460; font-size: 0.9em;'>
                💡 <strong>Tip:</strong> The template includes 2 sample rows. Replace them with your data while keeping the exact column names.
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Upload Your CSV File",
        type=['csv'],
        help="Maximum file size: 200MB. Ensure all required columns are present."
    )
    
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            
            # Display preview
            st.markdown("""
            <div style='background: #e8f5e9; padding: 15px; border-radius: 10px; margin: 15px 0; border-left: 4px solid #2ecc71;'>
                <h4 style='color: #1b5e20; margin: 0;'>✓ File Uploaded Successfully</h4>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"**Preview of uploaded data** ({len(df)} rows found):")
            st.dataframe(df.head(10), use_container_width=True)
            
            if len(df) > 10:
                st.info(f"Showing first 10 rows. Total rows to process: {len(df)}")
            
            if st.button("🚀 Process Batch Predictions", use_container_width=True, type="primary"):
                results = []
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for idx, row in df.iterrows():
                    status_text.text(f"Processing row {idx + 1} of {len(df)}...")
                    
                    try:
                        row_dict = row.to_dict()
                        processed = preprocess_input(row_dict)
                        prediction = model.predict(processed)[0]
                        probabilities = model.predict_proba(processed)[0]
                        
                        predicted_class = class_names[prediction]
                        confidence = float(probabilities[prediction])
                        
                        # Get recommendation data
                        rec_data = get_recommendations(predicted_class, row_dict)
                        
                        results.append({
                            'Row_Number': idx + 1,
                            'Prediction': predicted_class.replace('_', ' '),
                            'Risk_Level': rec_data['risk_level'],
                            'Confidence': f"{confidence*100:.1f}%",
                            'Status': '✓ Success'
                        })
                    except Exception as e:
                        results.append({
                            'Row_Number': idx + 1,
                            'Prediction': 'Error',
                            'Risk_Level': '-',
                            'Confidence': '-',
                            'Status': f'✗ Failed: {str(e)}'
                        })
                    
                    progress_bar.progress((idx + 1) / len(df))
                
                status_text.empty()
                
                # Create results dataframe
                results_df = pd.DataFrame(results)
                
                # Merge with original data
                df_with_results = df.copy()
                df_with_results['Row_Number'] = range(1, len(df) + 1)
                df_with_results = df_with_results.merge(results_df, on='Row_Number', how='left')
                
                # Success message
                successful = len([r for r in results if 'Success' in r['Status']])
                failed = len(results) - successful
                
                st.markdown(f"""
                <div style='background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%); 
                            padding: 25px; border-radius: 15px; margin: 20px 0; 
                            border-left: 4px solid #28a745; box-shadow: 0 4px 15px rgba(40, 167, 69, 0.2);'>
                    <h3 style='color: #155724; margin: 0 0 15px 0;'>✓ Batch Processing Complete!</h3>
                    <div style='display: grid; grid-template-columns: repeat(3, 1fr); gap: 15px;'>
                        <div style='text-align: center; background: white; padding: 15px; border-radius: 10px;'>
                            <div style='font-size: 2em; color: #667eea; font-weight: bold;'>{len(df)}</div>
                            <div style='color: #666; font-size: 0.9em;'>Total Rows</div>
                        </div>
                        <div style='text-align: center; background: white; padding: 15px; border-radius: 10px;'>
                            <div style='font-size: 2em; color: #28a745; font-weight: bold;'>{successful}</div>
                            <div style='color: #666; font-size: 0.9em;'>Successful</div>
                        </div>
                        <div style='text-align: center; background: white; padding: 15px; border-radius: 10px;'>
                            <div style='font-size: 2em; color: #dc3545; font-weight: bold;'>{failed}</div>
                            <div style='color: #666; font-size: 0.9em;'>Failed</div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Display results
                st.markdown("### 📊 Complete Results with Input Data")
                st.dataframe(df_with_results, use_container_width=True, height=400)
                
                # Download section
                st.markdown("### 💾 Download Options")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Full results with input data
                    csv_full = df_with_results.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Complete Results (Input + Predictions)",
                        data=csv_full,
                        file_name=f"batch_predictions_complete_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                        use_container_width=True,
                        help="Downloads all input data with prediction results"
                    )
                
                with col2:
                    # Results only
                    csv_results = results_df.to_csv(index=False)
                    st.download_button(
                        label="📊 Download Predictions Only",
                        data=csv_results,
                        file_name=f"batch_predictions_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                        use_container_width=True,
                        help="Downloads only the prediction results"
                    )
                
                # Summary by risk level
                if successful > 0:
                    st.markdown("### 📈 Summary by Risk Level")
                    risk_summary = df_with_results[df_with_results['Status'] == '✓ Success']['Risk_Level'].value_counts()
                    
                    fig = px.pie(
                        values=risk_summary.values,
                        names=risk_summary.index,
                        title="Distribution of Risk Levels",
                        color_discrete_sequence=px.colors.sequential.RdBu
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
        except Exception as e:
            st.error(f"Error reading CSV file: {str(e)}")
            st.info("Please ensure your CSV file matches the template format and contains all required columns.")
    
    else:
        st.markdown("""
        <div style='border: 2px dashed #667eea; border-radius: 15px; padding: 40px; 
                    text-align: center; background: #f8f9ff; margin: 20px 0;'>
            <h3 style='color: #667eea; margin-bottom: 10px;'>📂 Ready to Upload</h3>
            <p style='color: #666; margin: 0;'>Click "Browse files" above to select your CSV file</p>
            <p style='color: #999; font-size: 0.9em; margin-top: 10px;'>Supported format: CSV | Maximum size: 200MB</p>
        </div>
        """, unsafe_allow_html=True)
