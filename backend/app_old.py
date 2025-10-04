"""
FLASK BACKEND API
Web server that accepts input and returns predictions + recommendations
Bridge between ML model and user interface
Output: REST API endpoints for predictions
"""
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import pandas as pd
import numpy as np
import pickle
import os
from datetime import datetime

app = Flask(__name__, static_folder='../frontend', template_folder='../frontend')
CORS(app)

# Global variables
model = None
preprocessor = None
class_names = None

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
    """Preprocess input data exactly like training"""
    try:
        df = pd.DataFrame([data])
        
        # Feature Engineering (same as training)
        df['BMI'] = df['Weight'] / (df['Height'] ** 2)
        
        # Age Group
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
        
        # Activity Level - Updated for 0-7 days
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
        
        # Healthy Eating Score
        df['HealthyEatingScore'] = df['FCVC'] + df['NCP'] + (df['CH2O'] / 3)
        
        # High Tech Use - Updated for 0-12 hours
        df['HighTechUse'] = (df['TUE'] > 4).astype(int)  # Changed from 2 to 4
        
        # Active Transport
        df['ActiveTransport'] = df['MTRANS'].isin(['Walking', 'Bike']).astype(int)
        
        # Calorie Vegetable Ratio
        favc_val = 1 if str(df['FAVC'].values[0]).lower() in ['yes', '1'] else 0
        df['CalorieVegetableRatio'] = favc_val / (df['FCVC'] + 0.1)
        
        # Water Intake - Updated for 0.5-5L range
        ch2o_val = df['CH2O'].values[0]
        if ch2o_val <= 1.5:
            water = 'Low'
        elif ch2o_val <= 3:
            water = 'Medium'
        else:
            water = 'High'
        df['WaterIntake'] = water
        
        # Unhealthy Habits
        calc_val = 0 if str(df['CALC'].values[0]).lower() == 'no' else 1
        smoke_val = 1 if str(df['SMOKE'].values[0]).lower() in ['yes', '1'] else 0
        df['UnhealthyHabits'] = calc_val + smoke_val
        
        # Encode categorical variables
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if col in preprocessor['label_encoders']:
                le = preprocessor['label_encoders'][col]
                val = str(df[col].values[0])
                if val not in le.classes_:
                    val = le.classes_[0]
                df[col] = le.transform([val])
        
        # Scale numerical features
        numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        df[numerical_cols] = preprocessor['scaler'].transform(df[numerical_cols])
        
        # Ensure all required features present
        for feature in preprocessor['feature_names']:
            if feature not in df.columns:
                df[feature] = 0
        
        df = df[preprocessor['feature_names']]
        return df
    
    except Exception as e:
        raise Exception(f"Preprocessing error: {str(e)}")

def get_recommendations(predicted_class, user_data):
    """Generate personalized recommendations based on prediction and user data"""
    
    # Base recommendations by obesity class
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
    
    # Get base info
    info = base_recommendations.get(predicted_class, base_recommendations['Normal_Weight'])
    
    # Personalized recommendations based on user data
    personalized = []
    
    # Physical Activity - Updated for 0-7 range
    faf_val = user_data.get('FAF', 0)
    if faf_val < 2:
        personalized.append({
            'category': '🏃 Physical Activity',
            'advice': f"Your current activity level is very low ({faf_val} days/week). Aim to increase to at least 3-5 days per week with 30 minutes of moderate exercise."
        })
    elif faf_val < 3:
        personalized.append({
            'category': '🏃 Physical Activity',
            'advice': f"You exercise {faf_val} days/week. Try to increase to 4-6 days for optimal health benefits."
        })
    elif faf_val >= 5:
        personalized.append({
            'category': '🏃 Physical Activity',
            'advice': f"Excellent! You exercise {faf_val} days/week. Maintain this outstanding routine and ensure adequate recovery days."
        })
    else:
        personalized.append({
            'category': '🏃 Physical Activity',
            'advice': f"Good job! You exercise {faf_val} days/week. This is within the healthy range."
        })
    
    # Water Consumption - Updated for 0.5-5L range
    ch2o_val = user_data.get('CH2O', 0)
    if ch2o_val < 2:
        personalized.append({
            'category': '💧 Hydration',
            'advice': f"You drink {ch2o_val}L of water daily. Increase to at least 2-3L per day for better metabolism and health."
        })
    elif ch2o_val >= 3:
        personalized.append({
            'category': '💧 Hydration',
            'advice': f"Excellent! You drink {ch2o_val}L daily. Maintain these great hydration habits."
        })
    else:
        personalized.append({
            'category': '💧 Hydration',
            'advice': f"Good! You drink {ch2o_val}L daily. This is adequate for most people."
        })
    
    # Vegetable Consumption
    if user_data.get('FCVC', 0) < 2:
        personalized.append({
            'category': '🥗 Nutrition',
            'advice': f"Your vegetable intake (score: {user_data.get('FCVC', 0)}/3) needs improvement. Aim for 5+ servings of vegetables daily."
        })
    else:
        personalized.append({
            'category': '🥗 Nutrition',
            'advice': f"Good vegetable intake (score: {user_data.get('FCVC', 0)}/3). Continue eating a variety of colorful vegetables."
        })
    
    # Technology Use - Updated for 0-12 hour range
    tue_val = user_data.get('TUE', 0)
    if tue_val > 6:
        personalized.append({
            'category': '📱 Screen Time',
            'advice': f"You spend {tue_val} hours on devices daily. This is very high! Reduce screen time and replace with physical activities or social interactions."
        })
    elif tue_val > 4:
        personalized.append({
            'category': '📱 Screen Time',
            'advice': f"You spend {tue_val} hours on devices daily. Consider reducing to 3-4 hours maximum for better health."
        })
    elif tue_val <= 2:
        personalized.append({
            'category': '📱 Screen Time',
            'advice': f"Great! Your screen time of {tue_val} hours daily is within healthy limits."
        })
    
    # Transportation
    if user_data.get('MTRANS') in ['Automobile', 'Motorbike', 'Public_Transportation']:
        personalized.append({
            'category': '🚶 Active Transport',
            'advice': "Consider walking or biking for short distances instead of motorized transport to increase daily activity."
        })
    
    # High Caloric Food
    if str(user_data.get('FAVC', '')).lower() == 'yes':
        personalized.append({
            'category': '🍔 Diet Quality',
            'advice': "You frequently consume high-calorie foods. Replace with healthier alternatives like fruits, vegetables, and whole grains."
        })
    
    # Smoking
    if str(user_data.get('SMOKE', '')).lower() == 'yes':
        personalized.append({
            'category': '🚭 Smoking',
            'advice': "Smoking significantly impacts health and weight management. Seek professional help to quit smoking."
        })
    
    # Alcohol
    if user_data.get('CALC', 'no') != 'no':
        personalized.append({
            'category': '🍷 Alcohol',
            'advice': f"Alcohol consumption ({user_data.get('CALC')}) adds empty calories. Limit intake for better weight management."
        })
    
    # BMI-specific advice
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

@app.route('/')
def home():
    """Serve frontend"""
    return send_from_directory('../frontend', 'index.html')

@app.route('/api')
def api_info():
    """API information"""
    return jsonify({
        'message': 'Obesity Risk Prediction API with Personalized Recommendations',
        'version': '2.0',
        'endpoints': {
            '/api/predict': 'POST - Make prediction with recommendations',
            '/api/health': 'GET - Health check',
            '/api/classes': 'GET - Get all obesity classes'
        }
    })

@app.route('/api/health')
def health():
    """Health check"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/predict', methods=['POST'])
def predict():
    """Main prediction endpoint with recommendations"""
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
        
        # Get class name
        predicted_class = class_names[prediction]
        confidence = float(probabilities[prediction])
        
        # Get recommendations
        recommendations_data = get_recommendations(predicted_class, data)
        
        # Probability distribution
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
                for cls, prob in sorted_probs[:3]
            ],
            'recommendations': {
                'general': recommendations_data['general_recommendations'],
                'personalized': recommendations_data['personalized_recommendations']
            },
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify(response)
    
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

if __name__ == '__main__':
    print("="*70)
    print("OBESITY RISK PREDICTION API")
    print("With Personalized Health Recommendations")
    print("="*70)
    
    if load_model_and_preprocessor():
        print("\n✓ API ready to serve predictions!")
        print("✓ Personalized recommendations enabled!")
        print("="*70)
        print("\nStarting server on http://localhost:5000")
        print(" Open frontend/index.html in browser to use the app")
        print("\n  Keep this terminal running!")
        print("="*70 + "\n")
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("\n✗ Failed to load model/preprocessor")
        print("Please run data_preprocessing.py and model_training.py first")