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
from io import BytesIO
import plotly.graph_objects as go
import plotly.express as px

# PDF Generation
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak
from reportlab.lib.units import inch

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
    .sub-header {
        font-size: 1.5rem;
        color: #764ba2;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .info-box {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #d4edda;
        border-left: 5px solid #28a745;
    }
    .warning-box {
        background-color: #fff3cd;
        border-left: 5px solid #ffc107;
    }
    .danger-box {
        background-color: #f8d7da;
        border-left: 5px solid #dc3545;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        margin: 0.5rem 0;
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

def get_statistics():
    """Get usage statistics from database"""
    try:
        conn = sqlite3.connect('predictions.db')
        c = conn.cursor()
        
        total = c.execute('SELECT COUNT(*) FROM predictions').fetchone()[0]
        
        class_dist = c.execute('''SELECT predicted_class, COUNT(*) as count
                                 FROM predictions 
                                 GROUP BY predicted_class
                                 ORDER BY count DESC''').fetchall()
        
        avg_confidence = c.execute('SELECT AVG(confidence) FROM predictions').fetchone()[0]
        avg_bmi = c.execute('SELECT AVG(bmi) FROM predictions').fetchone()[0]
        
        risk_dist = c.execute('''SELECT risk_level, COUNT(*) as count
                                FROM predictions 
                                GROUP BY risk_level
                                ORDER BY count DESC''').fetchall()
        
        recent = c.execute('''SELECT timestamp, predicted_class, confidence, bmi
                             FROM predictions 
                             ORDER BY timestamp DESC 
                             LIMIT 10''').fetchall()
        
        conn.close()
        
        return {
            'total_predictions': total or 0,
            'class_distribution': dict(class_dist) if class_dist else {},
            'risk_distribution': dict(risk_dist) if risk_dist else {},
            'avg_confidence': round(avg_confidence, 2) if avg_confidence else 0,
            'avg_bmi': round(avg_bmi, 2) if avg_bmi else 0,
            'recent_predictions': recent or []
        }
    except Exception as e:
        st.error(f"Error getting statistics: {e}")
        return {
            'total_predictions': 0,
            'class_distribution': {},
            'risk_distribution': {},
            'avg_confidence': 0,
            'avg_bmi': 0,
            'recent_predictions': []
        }

# ============================================================================
# PDF GENERATION
# ============================================================================

def generate_health_report(prediction_data, user_data):
    """Generate comprehensive PDF health report"""
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, topMargin=0.5*inch)
    story = []
    styles = getSampleStyleSheet()
    
    # Custom Styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.HexColor('#667eea'),
        spaceAfter=20,
        alignment=1
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=16,
        textColor=colors.HexColor('#667eea'),
        spaceAfter=12,
        spaceBefore=12
    )
    
    # Title
    story.append(Paragraph("🏥 Obesity Risk Health Report", title_style))
    story.append(Spacer(1, 0.2*inch))
    
    # Date
    date_text = f"<b>Generated:</b> {datetime.now().strftime('%B %d, %Y at %I:%M %p')}"
    story.append(Paragraph(date_text, styles['Normal']))
    story.append(Spacer(1, 0.3*inch))
    
    # User Info
    story.append(Paragraph("👤 Personal Information", heading_style))
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
    story.append(Paragraph("📊 Assessment Results", heading_style))
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

def get_recommendations(predicted_class, user_data):
    """Generate personalized recommendations"""
    
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
                'Track
