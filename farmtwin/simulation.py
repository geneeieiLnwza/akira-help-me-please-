"""
FarmTwin v2 — Simulation Layer (Digital Twin Core 🔥)
Implements what-if analysis, scenario engine, and time simulation.
"""
import numpy as np
import pandas as pd


# ═══════════════════════════════════════════════════════════════════
# 1. WHAT-IF SIMULATION
# ═══════════════════════════════════════════════════════════════════

def simulate(model, encoder, scaler, base_params, changes=None):
    """
    Run a what-if simulation.
    base_params: dict of current farm conditions
    changes: dict of parameter changes (e.g., {'Rainfall_mm': '+10%', 'N_Fertilizer': -20})
    Returns: baseline_yield, simulated_yield, difference
    """
    # Build baseline input
    base_df = _build_input(base_params, encoder, scaler)
    baseline_yield = model.predict(base_df)[0]

    if changes is None:
        return baseline_yield, baseline_yield, 0.0

    # Apply changes
    modified_params = base_params.copy()
    for key, value in changes.items():
        if isinstance(value, str) and value.endswith('%'):
            pct = float(value.strip('%')) / 100
            modified_params[key] = modified_params[key] * (1 + pct)
        else:
            modified_params[key] = modified_params[key] + float(value)

    # Predict with changes
    mod_df = _build_input(modified_params, encoder, scaler)
    simulated_yield = model.predict(mod_df)[0]

    diff = simulated_yield - baseline_yield
    return max(0, baseline_yield), max(0, simulated_yield), diff


def _build_input(params, encoder, scaler):
    """Build a model-ready input DataFrame from raw parameters."""
    # Categorical columns
    cat_cols = ['Crop_Type', 'Soil_Type', 'Season', 'Location']
    cat_data = {c: [params[c]] for c in cat_cols}
    encoded = encoder.transform(pd.DataFrame(cat_data))
    encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(cat_cols))

    # Numeric columns
    numeric_cols = ['Temperature_C', 'Rainfall_mm', 'Humidity_pct', 'Soil_Moisture_pct',
                    'Irrigation_mm', 'N_Fertilizer', 'P_Fertilizer', 'K_Fertilizer']
    num_data = {c: [params[c]] for c in numeric_cols}
    num_df = pd.DataFrame(num_data)

    # Feature engineering
    num_df['Total_Water'] = num_df['Rainfall_mm'] + num_df['Irrigation_mm']
    num_df['Total_NPK'] = num_df['N_Fertilizer'] + num_df['P_Fertilizer'] + num_df['K_Fertilizer']
    num_df['N_Ratio'] = num_df['N_Fertilizer'] / (num_df['Total_NPK'] + 1)

    # Scale
    scale_cols = ['Temperature_C', 'Rainfall_mm', 'Humidity_pct', 'Soil_Moisture_pct',
                  'Irrigation_mm', 'N_Fertilizer', 'P_Fertilizer', 'K_Fertilizer',
                  'Total_Water', 'Total_NPK', 'N_Ratio']
    num_df[scale_cols] = scaler.transform(num_df[scale_cols])

    # Combine
    return pd.concat([num_df, encoded_df], axis=1)


# ═══════════════════════════════════════════════════════════════════
# 2. SCENARIO ENGINE
# ═══════════════════════════════════════════════════════════════════

PREDEFINED_SCENARIOS = {
    'best_case': {
        'name': '🌟 Best Case (Optimal Conditions)',
        'changes': {'Rainfall_mm': '+15%', 'Irrigation_mm': '+20%', 'N_Fertilizer': '+10%'}
    },
    'worst_case': {
        'name': '⚠️ Worst Case (Drought + Low Fertilizer)',
        'changes': {'Rainfall_mm': '-40%', 'Irrigation_mm': '-30%', 'N_Fertilizer': '-25%'}
    },
    'drought': {
        'name': '🔥 Drought Scenario',
        'changes': {'Rainfall_mm': '-50%', 'Humidity_pct': '-20%'}
    },
    'flood': {
        'name': '🌊 Flood Scenario',
        'changes': {'Rainfall_mm': '+80%', 'Soil_Moisture_pct': '+40%'}
    },
    'organic': {
        'name': '🌿 Organic Farming (Low Fertilizer)',
        'changes': {'N_Fertilizer': '-50%', 'P_Fertilizer': '-40%', 'K_Fertilizer': '-40%'}
    },
    'intensive': {
        'name': '⚡ Intensive Farming (High Input)',
        'changes': {'Irrigation_mm': '+50%', 'N_Fertilizer': '+40%', 'P_Fertilizer': '+30%', 'K_Fertilizer': '+30%'}
    },
}


def run_scenario(model, encoder, scaler, base_params, scenario_type='best_case'):
    """Run a predefined scenario."""
    scenario = PREDEFINED_SCENARIOS.get(scenario_type, PREDEFINED_SCENARIOS['best_case'])
    baseline, simulated, diff = simulate(model, encoder, scaler, base_params, scenario['changes'])
    return {
        'scenario': scenario['name'],
        'baseline_yield': round(baseline, 2),
        'simulated_yield': round(simulated, 2),
        'difference': round(diff, 2),
        'change_pct': round((diff / (baseline + 1)) * 100, 2)
    }


def run_all_scenarios(model, encoder, scaler, base_params):
    """Run all predefined scenarios and return comparison table."""
    results = []
    for key in PREDEFINED_SCENARIOS:
        result = run_scenario(model, encoder, scaler, base_params, key)
        results.append(result)
    return pd.DataFrame(results)


# ═══════════════════════════════════════════════════════════════════
# 3. TIME SIMULATION
# ═══════════════════════════════════════════════════════════════════

def predict_future(model, encoder, scaler, base_params, years_ahead=5):
    """
    Simulate future yields by adjusting climate trends.
    Assumes gradual warming + rainfall variability over time.
    """
    results = []
    current_year = base_params.get('Year', 2024)

    for y in range(years_ahead):
        future_year = current_year + y
        # Climate change trends (simplified)
        temp_increase = y * 0.15  # +0.15°C per year
        rain_change_pct = np.random.uniform(-5, 5)  # random ±5% rainfall

        changes = {
            'Temperature_C': temp_increase,
            'Rainfall_mm': f'{rain_change_pct:.1f}%'
        }

        baseline, simulated, diff = simulate(model, encoder, scaler, base_params, changes)
        results.append({
            'Year': future_year,
            'Predicted_Yield': round(simulated, 2),
            'Temp_Change': f'+{temp_increase:.2f}°C',
            'Rain_Change': f'{rain_change_pct:+.1f}%'
        })

    return pd.DataFrame(results)
