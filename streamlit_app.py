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
    }
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .user-message {
        background-color: #e8f4f8;
        text-align: right;
    }
    .assistant-message {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
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
    page = st.radio("Select Page", ["Single Prediction", "AI Health Assistant", "Batch Upload", "Statistics"])
    
    st.markdown("---")
    st.info("This is an AI assessment tool. Always consult healthcare professionals for medical advice.")

if page == "Single Prediction":
    st.header("Enter Your Health Information")
    
    with st.form("prediction_form"):
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
            
            # Recommendations
            st.subheader("General Health Recommendations")
            for rec in recommendations_data['general_recommendations']:
                st.write(f"✓ {rec}")
            
            st.subheader("Personalized Recommendations")
            for rec in recommendations_data['personalized_recommendations']:
                with st.expander(rec['category']):
                    st.write(rec['advice'])
            
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
    
    st.info("👋 Welcome! I'll guide you through a friendly conversation to assess your obesity risk.")
    
    # Display chat messages
    for msg in st.session_state.chat_messages:
        if msg['role'] == 'assistant':
            st.markdown(f'<div class="chat-message assistant-message">🤖 {msg["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="chat-message user-message">👤 {msg["content"]}</div>', unsafe_allow_html=True)
    
    # Current question
    if st.session_state.chat_step < len(CONVERSATION_FLOW):
        current_q = CONVERSATION_FLOW[st.session_state.chat_step]
        
        st.markdown(f'<div class="chat-message assistant-message">🤖 {current_q["question"]}</div>', unsafe_allow_html=True)
        
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
            if st.button("Submit Answer", use_container_width=True):
                st.session_state.chat_messages.append({'role': 'user', 'content': str(user_input)})
                st.session_state.chat_data[current_q['field']] = user_input
                st.session_state.chat_step += 1
                st.rerun()
        
        with col2:
            if st.button("Start Over", use_container_width=True):
                st.session_state.chat_step = 0
                st.session_state.chat_data = {}
                st.session_state.chat_messages = []
                st.rerun()
    
    else:
        # All questions answered - make prediction
        st.success("Assessment Complete! Analyzing your data...")
        
        with st.spinner("Processing..."):
            processed_data = preprocess_input(st.session_state.chat_data)
            prediction = model.predict(processed_data)[0]
            probabilities = model.predict_proba(processed_data)[0]
            
            predicted_class = class_names[prediction]
            confidence = float(probabilities[prediction])
            
            recommendations_data = get_recommendations(predicted_class, st.session_state.chat_data)
            
            # Display Results
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Classification", recommendations_data['status'])
            with col2:
                st.metric("Risk Level", recommendations_data['risk_level'])
            with col3:
                st.metric("Confidence", f"{confidence*100:.1f}%")
            
            # Recommendations
            st.subheader("Your Health Recommendations")
            for rec in recommendations_data['general_recommendations']:
                st.write(f"✓ {rec}")
            
            st.subheader("Personalized Advice")
            for rec in recommendations_data['personalized_recommendations']:
                with st.expander(rec['category']):
                    st.write(rec['advice'])
            
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
                label="Download PDF Report",
                data=pdf_buffer,
                file_name=f"health_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                mime="application/pdf"
            )
            
            if st.button("Start New Assessment"):
                st.session_state.chat_step = 0
                st.session_state.chat_data = {}
                st.session_state.chat_messages = []
                st.rerun()

elif page == "Batch Upload":
    st.header("Batch Predictions from CSV")
    
    st.info("Upload a CSV file with health data for multiple predictions")
    
    # Template download
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
        label="Download CSV Template",
        data=csv,
        file_name="obesity_prediction_template.csv",
        mime="text/csv"
    )
    
    uploaded_file = st.file_uploader("Upload CSV File", type=['csv'])
    
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        
        if st.button("Process Batch"):
            results = []
            progress_bar = st.progress(0)
            
            for idx, row in df.iterrows():
                try:
                    row_dict = row.to_dict()
                    processed = preprocess_input(row_dict)
                    prediction = model.predict(processed)[0]
                    probabilities = model.predict_proba(processed)[0]
                    
                    predicted_class = class_names[prediction]
                    confidence = float(probabilities[prediction])
                    
                    results.append({
                        'Row': idx + 1,
                        'Prediction': predicted_class.replace('_', ' '),
                        'Confidence': f"{confidence*100:.1f}%",
                        'Status': 'Success'
                    })
                except Exception as e:
                    results.append({
                        'Row': idx + 1,
                        'Prediction': 'Error',
                        'Confidence': '-',
                        'Status': f'Failed: {str(e)}'
                    })
                
                progress_bar.progress((idx + 1) / len(df))
            
            results_df = pd.DataFrame(results)
            st.success(f"Processed {len(df)} rows!")
            st.dataframe(results_df, use_container_width=True)
            
            csv_results = results_df.to_csv(index=False)
            st.download_button(
                label="Download Results",
                data=csv_results,
                file_name=f"batch_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )

else:  # Statistics
    st.header("System Statistics")
    
    if len(st.session_state.predictions_history) > 0:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Predictions", len(st.session_state.predictions_history))
        
        with col2:
            avg_conf = np.mean([p['confidence'] for p in st.session_state.predictions_history])
            st.metric("Average Confidence", f"{avg_conf*100:.1f}%")
        
        with col3:
            avg_bmi = np.mean([p['bmi'] for p in st.session_state.predictions_history])
            st.metric("Average BMI", f"{avg_bmi:.2f}")
        
        # Class distribution
        class_counts = pd.DataFrame(st.session_state.predictions_history)['class'].value_counts()
        fig = px.pie(values=class_counts.values, names=class_counts.index, 
                    title="Prediction Distribution")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No predictions yet. Make some predictions to see statistics!")
