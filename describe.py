import pandas as pd
import math

def count(data, Feature):
	feature = data[Feature].dropna().tolist()
	return len(feature)

def mean(data, Feature):
	feature = data[Feature].dropna().tolist()
	if not feature:
		return float('nan')
	mean_value = sum(feature) / len(feature)
	return mean_value

def std(data, Feature):
	feature = data[Feature].dropna().tolist()
	if len(feature) < 2:
		return float('nan')
	mean_value = mean(data, Feature)
	variance = sum((x - mean_value) ** 2 for x in feature) / len(feature)
	std_deviation = math.sqrt(variance)
	return std_deviation

def min(data, Feature):
	sorted_data = data.sort_values(by=Feature)
	return sorted_data[Feature].dropna().iloc[0]

def quartile(data, Feature):
	sorted_data = data.sort_values(by=Feature)
	return sorted_data[Feature].dropna().iloc[int(len(sorted_data) * 0.25)]

def mediane(data, Feature):
	sorted_data = data.sort_values(by=Feature)
	return sorted_data[Feature].dropna().iloc[int(len(sorted_data) * 0.50)]

def T_quartile(data, Feature):
	sorted_data = data.sort_values(by=Feature)
	return sorted_data[Feature].dropna().iloc[int(len(sorted_data) * 0.75)]

def max(data, Feature):
	sorted_data = data.sort_values(by=Feature)
	return sorted_data[Feature].dropna().iloc[-1]

def main():
	data = pd.read_csv('datasets/dataset_train.csv')

	def display(f, data, features):
		results = [f(data, feature) for feature in features]
		formatted_results = [f"{result:>16.6f}"[:16] for result in results]
		print(f"{f.__name__:<10}:\t{' '.join(formatted_results)}")

	all_features = data.columns.tolist()
	all_features = all_features[6:]

	for i in range(0, len(all_features), 4):
		features = all_features[i:i + 4]
		print(f"\n\t\t{' '.join([f'{feature[:16].strip():>16}' for feature in features])}")
		display(count, data, features)
		display(mean, data, features)
		display(std, data, features)
		display(min, data, features)
		display(quartile, data, features)
		display(mediane, data, features)
		display(T_quartile, data, features)
		display(max, data, features)

if __name__ == "__main__":
	try:
		main()
	except Exception as error:
		print(error)
