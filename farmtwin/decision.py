"""
FarmTwin v2 — Decision Support Layer
Provides farm management recommendations based on model predictions.
"""
import numpy as np
import pandas as pd
from farmtwin.simulation import simulate


# ═══════════════════════════════════════════════════════════════════
# 1. FERTILIZER RECOMMENDATION
# ═══════════════════════════════════════════════════════════════════

def recommend_fertilizer(model, encoder, scaler, base_params, n_range=(20, 250, 10)):
    """
    Find the optimal Nitrogen fertilizer level by simulating across a range.
    Returns: recommended N level + yield curve data for visualization.
    """
    start, stop, step = n_range
    results = []

    for n_val in range(start, stop, step):
        test_params = base_params.copy()
        test_params['N_Fertilizer'] = n_val
        baseline, predicted, _ = simulate(model, encoder, scaler, test_params)
        results.append({'N_Fertilizer': n_val, 'Predicted_Yield': round(predicted, 2)})

    df = pd.DataFrame(results)
    best = df.loc[df['Predicted_Yield'].idxmax()]

    recommendation = {
        'optimal_N': int(best['N_Fertilizer']),
        'expected_yield': best['Predicted_Yield'],
        'current_N': base_params.get('N_Fertilizer', 0),
        'curve_data': df
    }

    diff = recommendation['optimal_N'] - recommendation['current_N']
    if diff > 10:
        recommendation['advice'] = f"Recommend increasing N fertilizer from {recommendation['current_N']:.0f} to {recommendation['optimal_N']} kg/ha (+{diff:.0f})"
    elif diff < -10:
        recommendation['advice'] = f"Recommend decreasing N fertilizer from {recommendation['current_N']:.0f} to {recommendation['optimal_N']} kg/ha ({diff:.0f})"
    else:
        recommendation['advice'] = f"Current N level ({recommendation['current_N']:.0f}) is already near optimal"

    return recommendation


# ═══════════════════════════════════════════════════════════════════
# 2. CROP RECOMMENDATION
# ═══════════════════════════════════════════════════════════════════

def recommend_crop(model, encoder, scaler, base_params, crops=None):
    """
    Compare predicted yield across different crops given the same conditions.
    Recommends the highest-yielding crop.
    """
    if crops is None:
        crops = ['Rice', 'Wheat', 'Maize', 'Soybean']

    results = []
    for crop in crops:
        test_params = base_params.copy()
        test_params['Crop_Type'] = crop
        baseline, predicted, _ = simulate(model, encoder, scaler, test_params)
        results.append({'Crop': crop, 'Predicted_Yield': round(predicted, 2)})

    df = pd.DataFrame(results).sort_values('Predicted_Yield', ascending=False)
    best_crop = df.iloc[0]['Crop']
    best_yield = df.iloc[0]['Predicted_Yield']

    return {
        'recommended_crop': best_crop,
        'expected_yield': best_yield,
        'comparison': df,
        'advice': f"In these conditions, {best_crop} is recommended (expected yield: {best_yield:,.0f} kg/ha)"
    }


# ═══════════════════════════════════════════════════════════════════
# 3. RISK ASSESSMENT
# ═══════════════════════════════════════════════════════════════════

def assess_risk(model, encoder, scaler, base_params):
    """
    Assess farming risk by comparing best vs worst case scenarios.
    Returns risk level and recommendations.
    """
    from farmtwin.simulation import run_scenario

    best = run_scenario(model, encoder, scaler, base_params, 'best_case')
    worst = run_scenario(model, encoder, scaler, base_params, 'worst_case')
    baseline_yield = best['baseline_yield']

    yield_range = best['simulated_yield'] - worst['simulated_yield']
    volatility = (yield_range / (baseline_yield + 1)) * 100

    if volatility > 60:
        risk_level = 'HIGH RISK'
        recommendation = 'Consider adding irrigation systems and diversifying crop types to reduce risk.'
    elif volatility > 35:
        risk_level = 'MEDIUM RISK'
        recommendation = 'Monitor weather conditions closely and prepare contingency plans.'
    else:
        risk_level = 'LOW RISK'
        recommendation = 'Conditions are relatively stable. Proceed with normal operations.'

    return {
        'risk_level': risk_level,
        'volatility_pct': round(volatility, 2),
        'best_yield': best['simulated_yield'],
        'worst_yield': worst['simulated_yield'],
        'baseline_yield': baseline_yield,
        'recommendation': recommendation
    }
