"""
BACKEND
Predictions, PDF Reports, Database, Statistics, Batch Upload
"""

from flask import Flask, request, jsonify, send_from_directory, send_file
from flask_cors import CORS
import pandas as pd
import numpy as np
import pickle
import os
import sqlite3
from datetime import datetime
from io import BytesIO

# PDF Generation
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak
from reportlab.lib.units import inch

app = Flask(__name__, static_folder='../frontend', template_folder='../frontend')
CORS(app)

# Global variables
model = None
preprocessor = None
class_names = None

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
    print("✓ Database initialized")

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
        print(f"Error saving prediction: {e}")

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
            'recent_predictions': [
                {
                    'timestamp': r[0],
                    'class': r[1],
                    'confidence': round(r[2], 2),
                    'bmi': round(r[3], 2)
                } for r in recent
            ] if recent else []
        }
    except Exception as e:
        print(f"Error getting statistics: {e}")
        return {
            'total_predictions': 0,
            'class_distribution': {},
            'risk_distribution': {},
            'avg_confidence': 0,
            'avg_bmi': 0,
            'recent_predictions': []
        }

# ============================================================================
# PDF GENERATION FUNCTIONS
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
    
    # Date and Report Info
    date_text = f"<b>Generated:</b> {datetime.now().strftime('%B %d, %Y at %I:%M %p')}"
    story.append(Paragraph(date_text, styles['Normal']))
    story.append(Paragraph("<b>Report ID:</b> " + datetime.now().strftime('%Y%m%d%H%M%S'), styles['Normal']))
    story.append(Spacer(1, 0.3*inch))
    
    # User Information Section
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
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 11),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
        ('TOPPADDING', (0, 0), (-1, -1), 10),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey)
    ]))
    story.append(user_table)
    story.append(Spacer(1, 0.3*inch))
    
    # Assessment Results
    story.append(Paragraph("📊 Assessment Results", heading_style))
    
    prediction = prediction_data['prediction']
    result_data = [
        ['Classification:', prediction['status']],
        ['Risk Level:', prediction['risk_level']],
        ['Confidence Score:', f"{prediction['confidence']*100:.1f}%"],
    ]
    
    result_table = Table(result_data, colWidths=[2*inch, 4*inch])
    result_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#e8f4f8')),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 12),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
        ('TOPPADDING', (0, 0), (-1, -1), 12),
        ('GRID', (0, 0), (-1, -1), 1, colors.white)
    ]))
    story.append(result_table)
    story.append(Spacer(1, 0.3*inch))
    
    # Probability Distribution
    story.append(Paragraph("📈 Probability Distribution (Top 5)", heading_style))
    
    prob_data = [['Obesity Class', 'Probability']]
    for pred in prediction_data['top_predictions'][:5]:
        prob_data.append([
            pred['class'].replace('_', ' '),
            f"{pred['probability']*100:.1f}%"
        ])
    
    prob_table = Table(prob_data, colWidths=[4*inch, 2*inch])
    prob_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#667eea')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('ALIGN', (1, 0), (1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 11),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
        ('TOPPADDING', (0, 0), (-1, -1), 10),
        ('GRID', (0, 0), (-1, -1), 1, colors.grey),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f9f9f9')])
    ]))
    story.append(prob_table)
    story.append(PageBreak())
    
    # General Recommendations
    story.append(Paragraph("💡 General Health Recommendations", heading_style))
    story.append(Spacer(1, 0.1*inch))
    
    for i, rec in enumerate(prediction_data['recommendations']['general'], 1):
        bullet_text = f"<bullet>•</bullet> {rec}"
        story.append(Paragraph(bullet_text, styles['Normal']))
        story.append(Spacer(1, 0.08*inch))
    
    story.append(Spacer(1, 0.3*inch))
    
    # Personalized Recommendations
    story.append(Paragraph("🎯 Personalized Recommendations", heading_style))
    story.append(Spacer(1, 0.1*inch))
    
    for rec in prediction_data['recommendations']['personalized']:
        cat_style = ParagraphStyle('Category', parent=styles['Normal'], fontSize=11, textColor=colors.HexColor('#667eea'), fontName='Helvetica-Bold')
        story.append(Paragraph(rec['category'], cat_style))
        story.append(Paragraph(rec['advice'], styles['Normal']))
        story.append(Spacer(1, 0.12*inch))
    
    # Lifestyle Factors Summary
    story.append(PageBreak())
    story.append(Paragraph("📋 Your Lifestyle Factors", heading_style))
    
    lifestyle_data = [
        ['Factor', 'Your Value', 'Recommendation'],
        ['Physical Activity', f"{user_data.get('FAF', 0)} days/week", '3-5 days recommended'],
        ['Water Intake', f"{user_data.get('CH2O', 0)}L/day", '2-3L recommended'],
        ['Vegetable Consumption', f"{user_data.get('FCVC', 0)}/3", '2.5+ recommended'],
        ['Technology Use', f"{user_data.get('TUE', 0)} hours/day", '<4 hours recommended'],
        ['Main Meals', f"{user_data.get('NCP', 0)}/day", '3-4 meals recommended'],
    ]
    
    lifestyle_table = Table(lifestyle_data, colWidths=[2.5*inch, 1.8*inch, 2*inch])
    lifestyle_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#667eea')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
        ('TOPPADDING', (0, 0), (-1, -1), 10),
        ('GRID', (0, 0), (-1, -1), 1, colors.grey),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f9f9f9')])
    ]))
    story.append(lifestyle_table)
    
    # Disclaimer
    story.append(Spacer(1, 0.5*inch))
    disclaimer_style = ParagraphStyle(
        'Disclaimer',
        parent=styles['Normal'],
        fontSize=8,
        textColor=colors.grey,
        alignment=1,
        leading=10
    )
    story.append(Paragraph(
        "<b>IMPORTANT DISCLAIMER:</b><br/>"
        "This report is generated by an AI system and is for informational purposes only. "
        "It should NOT replace professional medical advice, diagnosis, or treatment. "
        "Always seek the advice of your physician or qualified health provider with any questions "
        "regarding your health condition. Never disregard professional medical advice or delay seeking it "
        "because of information in this report.",
        disclaimer_style
    ))
    
    # Footer
    story.append(Spacer(1, 0.2*inch))
    footer_style = ParagraphStyle('Footer', parent=styles['Normal'], fontSize=7, textColor=colors.grey, alignment=1)
    story.append(Paragraph(
        f"Obesity Risk Prediction System | Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | © 2025",
        footer_style
    ))
    
    # Build PDF
    doc.build(story)
    buffer.seek(0)
    return buffer

# ============================================================================
# ML MODEL FUNCTIONS
# ============================================================================

def load_model_and_preprocessor():
    """Load trained model and preprocessor"""
    global model, preprocessor, class_names
    
    try:
        with open('../models/best_model.pkl', 'rb') as f:
            model = pickle.load(f)
        print("✓ Model loaded successfully")
        
        with open('../models/preprocessor.pkl', 'rb') as f:
            preprocessor_data = pickle.load(f)
            preprocessor = {
                'scaler': preprocessor_data['scaler'],
                'label_encoders': preprocessor_data['label_encoders'],
                'target_encoder': preprocessor_data['target_encoder'],
                'feature_names': preprocessor_data['feature_names']
            }
            class_names = preprocessor_data['target_encoder'].classes_
        print("✓ Preprocessor loaded successfully")
        print(f"✓ Classes: {list(class_names)}")
        return True
    except Exception as e:
        print(f"✗ Error loading model: {str(e)}")
        return False

def preprocess_input(data):
    """Preprocess input data"""
    try:
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
    
    except Exception as e:
        raise Exception(f"Preprocessing error: {str(e)}")

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
    
    # Physical Activity
    faf_val = user_data.get('FAF', 0)
    if faf_val < 2:
        personalized.append({
            'category': '🏃 Physical Activity',
            'advice': f"Your activity level is low ({faf_val} days/week). Aim for 3-5 days/week with 30 min moderate exercise."
        })
    elif faf_val < 3:
        personalized.append({
            'category': '🏃 Physical Activity',
            'advice': f"You exercise {faf_val} days/week. Increase to 4-6 days for optimal benefits."
        })
    elif faf_val >= 5:
        personalized.append({
            'category': '🏃 Physical Activity',
            'advice': f"Excellent! {faf_val} days/week is outstanding. Ensure adequate recovery."
        })
    
    # Water
    ch2o_val = user_data.get('CH2O', 0)
    if ch2o_val < 2:
        personalized.append({
            'category': '💧 Hydration',
            'advice': f"You drink {ch2o_val}L daily. Increase to 2-3L for better health."
        })
    elif ch2o_val >= 3:
        personalized.append({
            'category': '💧 Hydration',
            'advice': f"Excellent! {ch2o_val}L daily is great hydration."
        })
    
    # Vegetables
    if user_data.get('FCVC', 0) < 2:
        personalized.append({
            'category': '🥗 Nutrition',
            'advice': f"Vegetable intake ({user_data.get('FCVC', 0)}/3) needs improvement. Aim for 5+ servings daily."
        })
    
    # Technology
    tue_val = user_data.get('TUE', 0)
    if tue_val > 6:
        personalized.append({
            'category': '📱 Screen Time',
            'advice': f"{tue_val} hours daily is very high! Reduce and add physical activities."
        })
    elif tue_val > 4:
        personalized.append({
            'category': '📱 Screen Time',
            'advice': f"{tue_val} hours daily. Consider reducing to 3-4 hours max."
        })
    
    # Transportation
    if user_data.get('MTRANS') in ['Automobile', 'Motorbike', 'Public_Transportation']:
        personalized.append({
            'category': '🚶 Active Transport',
            'advice': "Consider walking or biking for short distances to increase daily activity."
        })
    
    # High Caloric Food
    if str(user_data.get('FAVC', '')).lower() == 'yes':
        personalized.append({
            'category': '🍔 Diet Quality',
            'advice': "Frequent high-calorie foods detected. Replace with healthier alternatives."
        })
    
    # Smoking
    if str(user_data.get('SMOKE', '')).lower() == 'yes':
        personalized.append({
            'category': '🚭 Smoking',
            'advice': "Smoking impacts health significantly. Seek professional help to quit."
        })
    
    # BMI
    bmi = user_data.get('Weight', 0) / (user_data.get('Height', 1) ** 2) if user_data.get('Height', 0) > 0 else 0
    if bmi > 0:
        personalized.append({
            'category': '📊 BMI Analysis',
            'advice': f"Your BMI is {bmi:.1f}. " + (
                "This is in the healthy range." if 18.5 <= bmi < 25 else
                "This indicates underweight status." if bmi < 18.5 else
                "This indicates overweight status." if 25 <= bmi < 30 else
                "This indicates obesity. Medical consultation recommended."
            )
        })
    
    return {
        **info,
        'general_recommendations': info['general_advice'],
        'personalized_recommendations': personalized
    }

# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.route('/')
def home():
    """Serve frontend"""
    return send_from_directory('../frontend', 'index.html')

@app.route('/api')
def api_info():
    """API information"""
    return jsonify({
        'message': 'Obesity Risk Prediction API - Commercial Version',
        'version': '2.0',
        'features': ['Predictions', 'PDF Reports', 'Statistics', 'Batch Upload'],
        'endpoints': {
            '/api/predict': 'POST - Make prediction',
            '/api/download_report': 'POST - Download PDF report',
            '/api/predict_batch_csv': 'POST - Batch predictions from CSV',
            '/api/stats': 'GET - Get usage statistics',
            '/api/health': 'GET - Health check'
        }
    })

@app.route('/api/health')
def health():
    """Health check"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'database_connected': os.path.exists('predictions.db'),
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/predict', methods=['POST'])
def predict():
    """Main prediction endpoint"""
    try:
        if model is None or preprocessor is None:
            return jsonify({'error': 'Model not loaded'}), 500
        
        data = request.json
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Preprocess
        processed_data = preprocess_input(data)
        
        # Predict
        prediction = model.predict(processed_data)[0]
        probabilities = model.predict_proba(processed_data)[0]
        
        predicted_class = class_names[prediction]
        confidence = float(probabilities[prediction])
        
        recommendations_data = get_recommendations(predicted_class, data)
        
        prob_distribution = {
            class_names[i]: float(probabilities[i]) 
            for i in range(len(class_names))
        }
        
        sorted_probs = sorted(prob_distribution.items(), key=lambda x: x[1], reverse=True)
        
        response = {
            'success': True,
            'prediction': {
                'class': predicted_class,
                'confidence': confidence,
                'status': recommendations_data['status'],
                'risk_level': recommendations_data['risk_level'],
                'color': recommendations_data['color'],
                'icon': recommendations_data['icon']
            },
            'probabilities': prob_distribution,
            'top_predictions': [
                {'class': cls, 'probability': float(prob)} 
                for cls, prob in sorted_probs
            ],
            'recommendations': {
                'general': recommendations_data['general_recommendations'],
                'personalized': recommendations_data['personalized_recommendations']
            },
            'timestamp': datetime.now().isoformat()
        }
        
        # Save to database
        save_prediction(data, predicted_class, confidence, recommendations_data['risk_level'])
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

@app.route('/api/download_report', methods=['POST'])
def download_report():
    """Generate and download PDF report"""
    try:
        if model is None or preprocessor is None:
            return jsonify({'error': 'Model not loaded'}), 500
        
        data = request.json
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Get prediction
        processed_data = preprocess_input(data)
        prediction = model.predict(processed_data)[0]
        probabilities = model.predict_proba(processed_data)[0]
        
        predicted_class = class_names[prediction]
        confidence = float(probabilities[prediction])
        
        recommendations_data = get_recommendations(predicted_class, data)
        
        prob_distribution = {
            class_names[i]: float(probabilities[i]) 
            for i in range(len(class_names))
        }
        
        sorted_probs = sorted(prob_distribution.items(), key=lambda x: x[1], reverse=True)
        
        prediction_data = {
            'prediction': {
                'class': predicted_class,
                'confidence': confidence,
                'status': recommendations_data['status'],
                'risk_level': recommendations_data['risk_level'],
                'color': recommendations_data['color'],
                'icon': recommendations_data['icon']
            },
            'top_predictions': [
                {'class': cls, 'probability': float(prob)} 
                for cls, prob in sorted_probs
            ],
            'recommendations': {
                'general': recommendations_data['general_recommendations'],
                'personalized': recommendations_data['personalized_recommendations']
            }
        }
        
        # Generate PDF
        pdf_buffer = generate_health_report(prediction_data, data)
        
        return send_file(
            pdf_buffer,
            mimetype='application/pdf',
            as_attachment=True,
            download_name=f'health_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pdf'
        )
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

@app.route('/api/predict_batch_csv', methods=['POST'])
def predict_batch_csv():
    """Batch predictions from CSV file"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'Empty filename'}), 400
        
        if not file.filename.endswith('.csv'):
            return jsonify({'error': 'Please upload a CSV file'}), 400
        
        # Read CSV
        try:
            df = pd.read_csv(file)
        except Exception as e:
            return jsonify({'error': f'Error reading CSV: {str(e)}'}), 400
        
        # Validate required columns
        required_columns = ['Gender', 'Age', 'Height', 'Weight', 'family_history_with_overweight', 
                          'FAVC', 'FCVC', 'NCP', 'CAEC', 'SMOKE', 'CH2O', 'SCC', 
                          'FAF', 'TUE', 'CALC', 'MTRANS']
        
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            return jsonify({
                'error': f'Missing required columns: {", ".join(missing_columns)}. Please download the template.'
            }), 400
        
        if len(df) == 0:
            return jsonify({'error': 'CSV file is empty'}), 400
        
        if len(df) > 1000:
            return jsonify({'error': 'Maximum 1000 rows allowed per batch'}), 400
        
        results = []
        for idx, row in df.iterrows():
            try:
                row_dict = row.to_dict()
                processed = preprocess_input(row_dict)
                prediction = model.predict(processed)[0]
                probabilities = model.predict_proba(processed)[0]
                
                predicted_class = class_names[prediction]
                confidence = float(probabilities[prediction])
                
                results.append({
                    'row': idx + 1,
                    'prediction': predicted_class.replace('_', ' '),
                    'confidence': f"{confidence*100:.1f}%",
                    'status': 'success'
                })
                
                # Save to database
                save_prediction(row_dict, predicted_class, confidence, 'Batch')
                
            except Exception as e:
                results.append({
                    'row': idx + 1,
                    'error': str(e),
                    'status': 'failed'
                })
        
        return jsonify({
            'success': True,
            'total_rows': len(df),
            'successful': sum(1 for r in results if r['status'] == 'success'),
            'failed': sum(1 for r in results if r['status'] == 'failed'),
            'results': results
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

@app.route('/api/stats')
def get_stats():
    """Get usage statistics"""
    try:
        stats = get_statistics()
        return jsonify({
            'success': True,
            'statistics': stats
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

@app.route('/api/classes')
def get_classes():
    """Get all obesity classes"""
    if class_names is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    return jsonify({
        'classes': list(class_names),
        'total': len(class_names)
    })

# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    # Get port from environment variable for cloud deployment
    port = int(os.environ.get('PORT', 5000))
    # Never use debug=True in production
    app.run(debug=False, host='0.0.0.0', port=port)
    
    # Initialize database
    init_db()
    
    # Load ML model
    if load_model_and_preprocessor():
        print("\n✓ API ready to serve predictions!")
        print("="*70)
        print("\n Starting server on http://localhost:5000")
        print(" Open frontend/index.html in browser")
        print("\n  Keep this terminal running!")
        print("="*70 + "\n")
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("\n✗ Failed to load model/preprocessor")

        print("Please run data_preprocessing.py and model_training.py first")

