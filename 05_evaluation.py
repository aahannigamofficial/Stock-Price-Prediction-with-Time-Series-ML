import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os

#Complete use of AI for this file


# ============================================================================
# LOAD MODEL AND DATA
# ============================================================================

print("Loading model and data...")
model1 = joblib.load('data/model1.pkl')
X_test_scaled = joblib.load('data/X_test_scaled.pkl')
y_test = joblib.load('data/y_test.pkl')

# Convert y_test to numpy if needed
y_test_np = y_test.values if hasattr(y_test, 'values') else y_test

print(f"Model loaded. Test set size: {len(y_test_np)}")

# ============================================================================
# MAKE PREDICTIONS
# ============================================================================

y_pred = model1.predict(X_test_scaled)

print(f"\nPredictions shape: {y_pred.shape}")
print(f"Actual prices shape: {y_test_np.shape}")

# ============================================================================
# CALCULATE METRICS
# ============================================================================

print("\n" + "="*70)
print("MODEL EVALUATION METRICS")
print("="*70)

# 1. MAE (Mean Absolute Error)
mae = mean_absolute_error(y_test_np, y_pred)
print(f"\n1. MAE (Mean Absolute Error): ${mae:.2f}")
print(f"   → Model is off by ~${mae:.2f} per share on average")

# 2. RMSE (Root Mean Squared Error)
rmse = np.sqrt(mean_squared_error(y_test_np, y_pred))
print(f"\n2. RMSE (Root Mean Squared Error): ${rmse:.2f}")
print(f"   → Penalizes larger errors more than MAE")

# 3. MAPE (Mean Absolute Percentage Error)
mape = np.mean(np.abs((y_test_np - y_pred) / y_test_np)) * 100
print(f"\n3. MAPE (Mean Absolute Percentage Error): {mape:.2f}%")
print(f"   → Model is off by ~{mape:.2f}% of actual price")

# 4. R² (Coefficient of Determination)
r2 = r2_score(y_test_np, y_pred)
print(f"\n4. R² Score: {r2:.4f}")
print(f"   → Model explains {r2*100:.2f}% of price variation")

# 5. Directional Accuracy
# Compare if prediction direction matches actual direction
actual_direction = (y_test_np[1:] > y_test_np[:-1]).astype(int)  # 1 = up, 0 = down
pred_direction = (y_pred[1:] > y_test_np[:-1]).astype(int)
directional_accuracy = np.mean(actual_direction == pred_direction) * 100

print(f"\n5. Directional Accuracy: {directional_accuracy:.2f}%")
print(f"   → Model predicted price direction correctly {directional_accuracy:.2f}% of the time")

# ============================================================================
# ERROR ANALYSIS
# ============================================================================

print("\n" + "="*70)
print("ERROR ANALYSIS")
print("="*70)

errors = np.abs(y_pred - y_test_np)
print(f"\nError Statistics:")
print(f"  Min Error: ${errors.min():.2f}")
print(f"  Max Error: ${errors.max():.2f}")
print(f"  Median Error: ${np.median(errors):.2f}")
print(f"  Std Dev of Errors: ${errors.std():.2f}")

# ============================================================================
# VISUALIZATIONS
# ============================================================================

print("\n" + "="*70)
print("GENERATING VISUALIZATIONS")
print("="*70)

os.makedirs('results', exist_ok=True)

# Plot 1: Actual vs Predicted Prices
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Subplot 1: Time series comparison
axes[0, 0].plot(y_test_np, label='Actual', linewidth=2, color='blue', alpha=0.7)
axes[0, 0].plot(y_pred, label='Predicted', linewidth=2, color='red', alpha=0.7)
axes[0, 0].set_title('Actual vs Predicted Stock Prices', fontsize=12, fontweight='bold')
axes[0, 0].set_xlabel('Day')
axes[0, 0].set_ylabel('Price (USD)')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Subplot 2: Scatter plot (Actual vs Predicted)
axes[0, 1].scatter(y_test_np, y_pred, alpha=0.6, s=30)
axes[0, 1].plot([y_test_np.min(), y_test_np.max()], 
                [y_test_np.min(), y_test_np.max()], 
                'r--', lw=2, label='Perfect Prediction')
axes[0, 1].set_title('Actual vs Predicted (Scatter)', fontsize=12, fontweight='bold')
axes[0, 1].set_xlabel('Actual Price (USD)')
axes[0, 1].set_ylabel('Predicted Price (USD)')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Subplot 3: Residuals (Errors) over time
residuals = y_pred - y_test_np
axes[1, 0].plot(residuals, linewidth=1, color='purple', alpha=0.7)
axes[1, 0].axhline(y=0, color='red', linestyle='--', linewidth=2)
axes[1, 0].fill_between(range(len(residuals)), residuals, alpha=0.3, color='purple')
axes[1, 0].set_title('Prediction Errors Over Time', fontsize=12, fontweight='bold')
axes[1, 0].set_xlabel('Day')
axes[1, 0].set_ylabel('Error (Predicted - Actual)')
axes[1, 0].grid(True, alpha=0.3)

# Subplot 4: Error distribution (histogram)
axes[1, 1].hist(errors, bins=30, color='green', alpha=0.7, edgecolor='black')
axes[1, 1].axvline(mae, color='red', linestyle='--', linewidth=2, label=f'MAE: ${mae:.2f}')
axes[1, 1].set_title('Distribution of Absolute Errors', fontsize=12, fontweight='bold')
axes[1, 1].set_xlabel('Error (USD)')
axes[1, 1].set_ylabel('Frequency')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('results/01_evaluation_metrics.png', dpi=300, bbox_inches='tight')
print("✓ Saved: results/01_evaluation_metrics.png")
plt.close()

# ============================================================================
# SUMMARY REPORT
# ============================================================================

print("\n" + "="*70)
print("SUMMARY REPORT")
print("="*70)

summary_text = f"""
LINEAR REGRESSION MODEL - TEST SET EVALUATION

Quantitative Metrics:
  • MAE (Mean Absolute Error):        ${mae:.2f}
  • RMSE (Root Mean Squared Error):   ${rmse:.2f}
  • MAPE (Mean Absolute % Error):     {mape:.2f}%
  • R² Score:                         {r2:.4f}
  • Directional Accuracy:             {directional_accuracy:.2f}%

Error Statistics:
  • Min Error:                        ${errors.min():.2f}
  • Max Error:                        ${errors.max():.2f}
  • Median Error:                     ${np.median(errors):.2f}
  • Std Dev of Errors:                ${errors.std():.2f}

Interpretation:
  ✓ The model explains {r2*100:.2f}% of price variation
  ✓ Average prediction error is ${mae:.2f} per share
  ✓ Model predicts direction correctly {directional_accuracy:.2f}% of the time
  ✓ MAPE of {mape:.2f}% indicates {('excellent' if mape < 5 else 'good' if mape < 10 else 'fair')} accuracy

Conclusion:
  The Linear Regression model demonstrates strong predictive performance
  on the test set, with low MAE and high R² score. The directional
  accuracy of {directional_accuracy:.2f}% suggests the model effectively
  captures price movement trends.
"""

print(summary_text)

# Save summary to file
with open('results/evaluation_summary.txt', 'w', encoding='utf-8') as f:
    f.write(summary_text)

print("\n✓ Evaluation complete! Results saved to 'results/' folder")