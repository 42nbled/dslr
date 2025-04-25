from os.path import exists
import sys
import pandas as pd
import numpy as np
import argparse
from tqdm import tqdm
from pathlib import Path
from LogisticRegression import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier as OVAC
from sklearn.linear_model import LogisticRegression as SklearnLogisticRegression

def	batched():
	batch_size = 100
	for model, house in zip(models, houses):
		with tqdm(total=X_scaled.shape[0]) as progress:
			progress.set_description(house.ljust(10))
			for i in range(X_scaled.shape[0] // batch_size + 1):
				start = batch_size * i
				end = batch_size * (i + 1)
				if X_scaled[start:end].shape[0] != 0:
					progress.update(X_scaled[start:end].shape[0])
					model.fit(X_scaled[start:end], (y == house).astype(int)[start:end])

def	stochastic():
	bars = {house: tqdm(total=X_scaled.shape[0] * 1, desc=house.ljust(10)) for house in houses}
	for j in range(1):
		index = np.random.permutation(X_scaled.shape[0])
		X_shuffle = X_scaled[index]
		y_shuffle = y[index]
		for model, house in zip(models, houses):
			progress = bars[house]
			progress.set_description(house.ljust(10))
			for i in range(X_scaled.shape[0]):
				progress.update(1)
				model.fit(X_shuffle[i:i + 1], (y_shuffle == house).astype(int)[i:i + 1], 0.01)

		probabilities = np.column_stack([model.predict_proba(X_scaled) for model in models])
		predicted_houses = np.array(houses)[np.argmax(probabilities, axis=1)]
		accuracy = (predicted_houses == y.values).mean()
		print(f"accuracy at {j}: {accuracy * 100:.3f}%")

def	default():
	for model, house in zip(models, houses):
		with tqdm(total=X_scaled.shape[0]) as progress:
			progress.set_description(house.ljust(10))
			progress.update(X_scaled.shape[0])
			model.fit(X_scaled, (y == house).astype(int))

parser = argparse.ArgumentParser()
parser.add_argument('dataset')
parser.add_argument('-m', '--model', required=False, choices=['default', 'batched', 'stochastic'], default='default')
args = parser.parse_args()

if not Path(args.dataset).exists():
	print('No dataset found')
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

if args.model == "batched":
	batched()
elif args.model == "stochastic":
	stochastic()
elif args.model == "default":
	default()

path = Path('trained_model.npy')
path.parent.mkdir(parents=True, exist_ok=True)
with open(path, 'wb') as f:
	for model in models:
		model.save(f)

probabilities = np.column_stack([model.predict_proba(X_scaled) for model in models])
predicted_houses = np.array(houses)[np.argmax(probabilities, axis=1)]
accuracy = (predicted_houses == y.values).mean()
print(f"Model accuracy   : {accuracy * 100:.3f}%")

sk_model = OVAC(SklearnLogisticRegression(max_iter=1000))
sk_model.fit(X_scaled, y)
sk_pred = sk_model.predict(X_scaled)
sk_accuracy = (sk_pred == y.values).mean()#pyright:ignore
print(f"sklearn accuracy : {sk_accuracy * 100:.3f}%")
