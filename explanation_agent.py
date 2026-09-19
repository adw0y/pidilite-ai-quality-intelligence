import pandas as pd

def _build_prompt(batch_data, prediction, shap_drivers):
    prompt = "You are a polymer process expert at a Pidilite manufacturing plant.\n\n"
    
    prompt += "Batch Data Context:\n"
    for k, v in batch_data.items():
        prompt += f"- {k}: {v}\n"
        
    prompt += "\nPrediction Context:\n"
    for k, v in prediction.items():
        prompt += f"- {k}: {v}\n"
        
    prompt += "\nTop SHAP Drivers:\n"
    for d in shap_drivers:
        prompt += f"- Feature: {d.get('feature')}, Value: {d.get('value')}, SHAP: {d.get('shap_value')}, Direction: {d.get('direction')}\n"
        
    prompt += "\nTask: Provide a 2-3 sentence explanation that an operator can understand, explaining the risk factors, followed by 1 actionable recommendation."
    
    return prompt

def _rule_based_explanation(batch_data, prediction, shap_drivers):
    target = prediction.get('target_name', 'Quality')
    risk = prediction.get('risk_level', 'Unknown')
    pred_val = prediction.get('predicted_value', 'N/A')
    
    explanation = f"The predicted {target} is {pred_val} (Risk: {risk}). "
    
    if shap_drivers:
        top_driver = shap_drivers[0]
        explanation += f"The primary factor is {top_driver.get('feature')} having a value of {top_driver.get('value')}, which {top_driver.get('direction')} the risk. "
        
    explanation += "Recommendation: "
    if risk.lower() in ['high', 'elevated']:
        explanation += "Please review the batch parameters and consult the shift supervisor."
    else:
        explanation += "Proceed with standard operating procedures."
        
    return explanation

def _call_gemini(prompt, api_key):
    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.0-flash')
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error calling Gemini API: {str(e)}"

def explain_batch(batch_data, prediction, shap_drivers, api_key=None):
    """
    If api_key provided, calls Gemini API to generate natural language explanation.
    If not, falls back to rule-based template.
    """
    if api_key:
        prompt = _build_prompt(batch_data, prediction, shap_drivers)
        response = _call_gemini(prompt, api_key)
        if not response.startswith("Error"):
            return response
            
    return _rule_based_explanation(batch_data, prediction, shap_drivers)
