import pandas as pd
import math

def count(data, Feature):
    feature = data[Feature].dropna().tolist()
    return len(feature)

def mean(data, Feature):
    feature = data[Feature].dropna().tolist()
    return sum(feature) / len(feature)

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

    Feature_1 = 'Arithmancy'
    Feature_2 = 'Astronomy'
    Feature_3 = 'Herbology'
    Feature_4 = 'Defense Against the Dark Arts'

    def display(f, data):
        features = [Feature_1, Feature_2, Feature_3, Feature_4]
        results = [f(data, feature) for feature in features]
        formatted_results = [f"{result:>16.6f}"[:16] for result in results]
        print(f"{f.__name__:<8}:\t{' '.join(formatted_results)}")

    print(f"\t\t{Feature_1[:16].strip():>16} {Feature_2[:16].strip():>16} {Feature_3[:16].strip():>16} {Feature_4[:16].strip():>16}")
    display(count, data)
    display(mean, data)
    display(std, data)
    display(min, data)
    display(quartile, data)
    display(mediane, data)
    display(T_quartile, data)
    display(max, data)

if __name__ == "__main__":
    main()
