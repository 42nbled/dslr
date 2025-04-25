import sys
import pandas as pd
import numpy as np
import argparse
from pathlib import Path
from LogisticRegression import LogisticRegression

parser = argparse.ArgumentParser()
parser.add_argument('dataset')
parser.add_argument('trained_model')
args = parser.parse_args()

if not Path(args.dataset).exists():
	print('No dataset found')
	exit(1)
if not Path(args.trained_model).exists():
	print('No model found')
	exit(1)

df = pd.read_csv(args.dataset)

X = df[["Herbology", "Defense Against the Dark Arts", "Ancient Runes", "Astronomy"]]
y = df["Hogwarts House"]

X_imputed: np.ndarray = X.values.copy()#pyright:ignore
col_means = np.nanmean(X_imputed, axis=0)
X_imputed[np.isnan(X_imputed)] = np.take(col_means, np.where(np.isnan(X_imputed))[1])
X_scaled = (X_imputed - X_imputed.mean(axis=0)) / X_imputed.std(axis=0)

houses = ["Gryffindor", "Hufflepuff", "Ravenclaw", "Slytherin"]
models = [LogisticRegression(max_iter=1000) for _ in houses]

model = Path(args.trained_model)
if not model.exists():
	print('No trained model found')
	exit(1)
with open(model, 'rb') as f:
	for model in models:
		model.load(f)

probabilities = np.column_stack([model.predict_proba(X_scaled) for model in models])
predicted_houses = np.array(houses)[np.argmax(probabilities, axis=1)]

results = pd.DataFrame({
	'Index': range(0, predicted_houses.shape[0]),
	'Hogwarts House': predicted_houses
})

results.to_csv('house.csv', index=False)
