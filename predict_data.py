import pandas as pd
import numpy as np
import joblib
import argparse
from process_smhi import process_smhi

parser = argparse.ArgumentParser()
parser.add_argument('smhi_filename')
parser.add_argument('-m', "--model", required=True)
parser.add_argument('-o','--output', default='predicted_pv_output_with_time.csv')
args = parser.parse_args()
smhi_filename = args.smhi_filename
output_file = args.output
model_file = args.model

print("Loading model...")
fitted_reg = joblib.load(model_file)

print("Processing data...")
X, time = process_smhi(smhi_filename)

print("Predicting...")
X_pred = fitted_reg.predict(X)
X_pred_clipped = np.where(X_pred < 0.1, 0, X_pred)

result = pd.DataFrame({
    'time': time,
    'predicted_pv_output': X_pred_clipped
})

result.to_csv(output_file, index=False)
