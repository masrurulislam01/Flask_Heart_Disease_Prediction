import joblib
import numpy as np
from flask import Flask, request, render_template_string

# Load trained model
model = joblib.load("random_forest_model.pkl")

# Feature definitions with units added
FEATURES = [
    ("Age", "number", {"min": 1, "max": 120, "step": 1, "unit": "years"}),
    ("Sex", "select", {"Male": 1, "Female": 0}),
    ("Chest Pain Type", "select", {
        "Typical Angina": 0,
        "Atypical Angina": 1,
        "Non-anginal Pain": 2,
        "Asymptomatic": 3
    }),
    ("Resting Blood Pressure", "number", {"min": 50, "max": 250, "step": 1, "unit": "mm Hg"}),
    ("Serum Cholesterol", "number", {"min": 100, "max": 600, "step": 1, "unit": "mg/dl"}),
    ("Fasting Blood Sugar > 120 mg/dl", "select", {"Yes": 1, "No": 0}),
    ("Resting ECG", "select", {
        "Normal": 0,
        "ST-T Wave Abnormality": 1,
        "Left Ventricular Hypertrophy": 2
    }),
    ("Max Heart Rate Achieved", "number", {"min": 60, "max": 220, "step": 1, "unit": "bpm"}),
    ("Exercise Induced Angina", "select", {"Yes": 1, "No": 0}),
    ("ST Depression (oldpeak)", "number", {"min": 0, "max": 10, "step": 0.1}),
    ("Slope of Peak Exercise ST Segment", "select", {
        "Upsloping": 0,
        "Flat": 1,
        "Downsloping": 2
    }),
    ("Number of Major Vessels", "number", {"min": 0, "max": 3, "step": 1}),
    ("Thalassemia", "select", {
        "Normal": 1,
        "Fixed Defect": 2,
        "Reversible Defect": 3
    })
]

NUM_FEATURES = len(FEATURES)

# Enhanced Bootstrap HTML Template
html_template = """
<!doctype html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Heart Disease Risk Assessment</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.10.0/font/bootstrap-icons.css">
    <style>
        :root {
            --card-bg: rgba(255, 255, 255, 0.95);
            --primary-bg: #e9f7fe;
            --success-color: #28a745;
            --danger-color: #dc3545;
        }
        
        body {
            background: linear-gradient(135deg, #f5f7fa 0%, #e4efe9 100%);
            min-height: 100vh;
            padding: 20px;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }
        
        .header {
            background: linear-gradient(90deg, #1a6fc4 0%, #0d4a8a 100%);
            color: white;
            border-radius: 10px 10px 0 0;
            padding: 25px 0;
            margin-bottom: 0;
        }
        
        .card {
            border-radius: 15px;
            border: none;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1);
            overflow: hidden;
            margin-bottom: 30px;
        }
        
        .card-body {
            padding: 0;
        }
        
        .form-container {
            padding: 30px;
            background-color: var(--card-bg);
        }
        
        .feature-card {
            background: white;
            border-radius: 10px;
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.05);
            transition: transform 0.3s, box-shadow 0.3s;
            height: 100%;
        }
        
        .feature-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 8px 15px rgba(0,0,0,0.1);
        }
        
        .form-label {
            font-weight: 600;
            color: #2c3e50;
            margin-bottom: 8px;
        }
        
        .form-control, .form-select {
            border-radius: 8px;
            padding: 10px 15px;
            border: 2px solid #e1e5eb;
            transition: border-color 0.3s;
        }
        
        .form-control:focus, .form-select:focus {
            border-color: #1a6fc4;
            box-shadow: 0 0 0 0.25rem rgba(26, 111, 196, 0.25);
        }
        
        .input-group-text {
            background: #e9f7fe;
            border: 2px solid #e1e5eb;
            border-right: none;
            font-weight: 500;
        }
        
        .submit-btn {
            background: linear-gradient(90deg, #1a6fc4 0%, #0d4a8a 100%);
            border: none;
            padding: 12px 30px;
            font-size: 1.1rem;
            font-weight: 600;
            border-radius: 50px;
            width: 100%;
            max-width: 300px;
            transition: transform 0.3s, box-shadow 0.3s;
        }
        
        .submit-btn:hover {
            transform: translateY(-3px);
            box-shadow: 0 5px 15px rgba(26, 111, 196, 0.4);
        }
        
        .result-card {
            text-align: center;
            padding: 30px;
            border-radius: 15px;
            margin-top: 20px;
        }
        
        .no-risk {
            background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
            border: 2px solid var(--success-color);
        }
        
        .risk {
            background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%);
            border: 2px solid var(--danger-color);
        }
        
        .result-icon {
            font-size: 4rem;
            margin-bottom: 20px;
        }
        
        .no-risk-icon {
            color: var(--success-color);
        }
        
        .risk-icon {
            color: var(--danger-color);
        }
        
        .result-title {
            font-size: 1.8rem;
            font-weight: 700;
            margin-bottom: 10px;
        }
        
        .footer {
            text-align: center;
            color: #6c757d;
            padding: 20px;
            font-size: 0.9rem;
        }
        
        .feature-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        
        .developer-credit {
            position: fixed;
            bottom: 10px;
            right: 20px;
            background: rgba(255, 255, 255, 0.8);
            padding: 5px 15px;
            border-radius: 20px;
            font-weight: 500;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            z-index: 100;
        }
        
        @media (max-width: 768px) {
            .feature-grid {
                grid-template-columns: 1fr;
            }
            .developer-credit {
                position: static;
                margin-top: 20px;
                text-align: center;
            }
        }
        
        .number-input-container {
            position: relative;
        }
        
        .number-input-controls {
            position: absolute;
            right: 10px;
            top: 50%;
            transform: translateY(-50%);
            display: flex;
            flex-direction: column;
            gap: 2px;
        }
        
        .number-btn {
            width: 24px;
            height: 20px;
            display: flex;
            align-items: center;
            justify-content: center;
            background: #f1f3f5;
            border: 1px solid #dee2e6;
            border-radius: 4px;
            cursor: pointer;
            font-size: 10px;
        }
        
        .number-btn:hover {
            background: #e9ecef;
        }
    </style>
</head>
<body>
    <div class="container-fluid">
        <div class="row justify-content-center">
            <div class="col-12 col-xl-10">
                <div class="card">
                    <div class="header text-center">
                        <h1><i class="bi bi-heart-pulse"></i> Heart Disease Risk Assessment</h1>
                        <p class="lead mb-0">Complete the form below to evaluate your cardiovascular health</p>
                    </div>
                    
                    <div class="form-container">
                        <form method="post" id="prediction-form">
                            <div class="feature-grid">
                                {% for idx, (label, field_type, params) in feature_list %}
                                <div class="feature-card">
                                    <label class="form-label">{{ label }}</label>
                                    
                                    {% if field_type == 'number' %}
                                        <div class="input-group">
                                            <input type="number" name="f{{idx}}" class="form-control"
                                                id="input-{{idx}}"
                                                min="{{ params.min }}" max="{{ params.max }}" step="{{ params.step }}" 
                                                required
                                                value="{{ (params.min + params.max)/2 | round(1) if 'step' in params and params.step == 0.1 else (params.min + params.max)//2 }}">
                                            {% if 'unit' in params %}
                                                <span class="input-group-text">{{ params.unit }}</span>
                                            {% endif %}
                                        </div>
                                        <div class="form-text text-muted">Range: {{ params.min }} - {{ params.max }}</div>
                                    
                                    {% elif field_type == 'select' %}
                                        <select name="f{{idx}}" class="form-select" required>
                                            <option value="" disabled selected>Select an option</option>
                                            {% for option_label, option_value in params.items() %}
                                                <option value="{{ option_value }}">{{ option_label }}</option>
                                            {% endfor %}
                                        </select>
                                    {% endif %}
                                </div>
                                {% endfor %}
                            </div>
                            
                            <div class="text-center mt-4">
                                <button type="submit" class="btn submit-btn text-white">
                                    <i class="bi bi-calculator me-2"></i>Calculate Risk
                                </button>
                            </div>
                        </form>
                        
                        {% if prediction_text %}
                        <div class="result-card {{ 'no-risk' if 'No Heart' in prediction_text else 'risk' }}">
                            <i class="result-icon bi {{ 'bi-check-circle-fill no-risk-icon' if 'No Heart' in prediction_text else 'bi-exclamation-triangle-fill risk-icon' }}"></i>
                            <h2 class="result-title">{{ prediction_text }}</h2>
                            <p class="mb-0">
                                {% if 'No Heart' in prediction_text %}
                                    Low risk of cardiovascular disease detected
                                {% else %}
                                    Potential heart disease risk detected - Consult your doctor
                                {% endif %}
                            </p>
                        </div>
                        {% endif %}
                    </div>
                </div>
                
                <div class="footer">
                    <p class="mb-0">Note: This prediction is based on machine learning algorithms and should not replace professional medical advice.</p>
                    <p>Heart Disease Prediction System | Developed by Ayon, Ruman</p>
                </div>
            </div>
        </div>
    </div>
    
    <div class="developer-credit">
        Developed by Ayon, Ruman
    </div>

    <script>
        // Enhanced number input handling
        document.addEventListener('DOMContentLoaded', function() {
            // Create increment/decrement buttons for number inputs
            document.querySelectorAll('input[type="number"]').forEach(input => {
                const container = document.createElement('div');
                container.className = 'number-input-container';
                input.parentNode.insertBefore(container, input);
                container.appendChild(input);
                
                const controls = document.createElement('div');
                controls.className = 'number-input-controls';
                
                const upBtn = document.createElement('div');
                upBtn.className = 'number-btn';
                upBtn.innerHTML = '▲';
                upBtn.onclick = () => {
                    input.stepUp();
                    input.dispatchEvent(new Event('change'));
                };
                
                const downBtn = document.createElement('div');
                downBtn.className = 'number-btn';
                downBtn.innerHTML = '▼';
                downBtn.onclick = () => {
                    input.stepDown();
                    input.dispatchEvent(new Event('change'));
                };
                
                controls.appendChild(upBtn);
                controls.appendChild(downBtn);
                container.appendChild(controls);
            });
            
            // Keyboard navigation support
            document.querySelectorAll('input, select').forEach(input => {
                input.addEventListener('keydown', function(e) {
                    if (e.key === 'Enter') {
                        e.preventDefault();
                        const form = document.getElementById('prediction-form');
                        form.dispatchEvent(new Event('submit'));
                    }
                });
            });
        });
    </script>
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
</body>
</html>
"""

# Flask app
app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def predict():
    prediction_text = None

    if request.method == "POST":
        try:
            # Validate and convert inputs
            features = []
            for i, (label, field_type, params) in enumerate(FEATURES):
                value = request.form.get(f"f{i}")
                if not value:
                    raise ValueError(f"Missing value for {label}")
                    
                if field_type == "number":
                    value = float(value)
                    min_val = params["min"]
                    max_val = params["max"]
                    if not (min_val <= value <= max_val):
                        raise ValueError(f"{label} must be between {min_val} and {max_val}")
                else:
                    value = int(value)
                    
                features.append(value)

            features_array = np.array([features])
            prediction = model.predict(features_array)[0]
            prediction_text = "✅ No Heart Disease Risk Detected" if prediction == 0 else "⚠️ Heart Disease Risk Detected"
            
        except Exception as e:
            prediction_text = f"❌ Error: {str(e)}"

    return render_template_string(
        html_template, 
        feature_list=list(enumerate(FEATURES)), 
        prediction_text=prediction_text
    )

if __name__ == "__main__":
    app.run(debug=True)