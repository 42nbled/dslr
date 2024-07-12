import pandas as pd
import matplotlib.pyplot as plt
import describe as desc

def quartile(data, feature):
	sorted_data = data.sort_values(by=feature)
	length = len(sorted_data[feature].dropna())
	if length == 0:
		return float('nan')
	return sorted_data[feature].dropna().iloc[int(length * 0.25)], sorted_data[feature].dropna().iloc[int(length * 0.75)]

def mediane(data, feature):
	sorted_data = data.sort_values(by=feature)
	length = len(sorted_data[feature].dropna())
	if length == 0:
		return float('nan')
	return sorted_data[feature].dropna().iloc[int(length * 0.50)]

def main():
	data = pd.read_csv('datasets/dataset_test.csv')

	Feature_1 = 'Arithmancy'
	Feature_2 = 'Astronomy'
	Feature_3 = 'Herbology'
	Feature_4 = 'Defense Against the Dark Arts'

	fig, ax = plt.subplots(figsize=(10, 8))
	colors = ['skyblue', 'orange', 'green', 'red']
	bar_width = 0.2  # Width of the bars
	positions = list(range(len(['Below Q1', 'Q1-Median', 'Median-Q3', 'Above Q3'])))

	min_value = float('inf')
	max_value = float('-inf')

	for i, feature in enumerate([Feature_1, Feature_2, Feature_3, Feature_4]):
		col_data = data[feature].dropna()

		Q1, Q3 = quartile(data, feature)
		median = mediane(data, feature)

		# Calculate counts in each quartile category
		below_Q1 = (col_data < Q1).sum()
		between_Q1_median = ((col_data >= Q1) & (col_data < median)).sum()
		between_median_Q3 = ((col_data >= median) & (col_data < Q3)).sum()
		above_Q3 = (col_data >= Q3).sum()

		counts = [below_Q1, between_Q1_median, between_median_Q3, above_Q3]
		labels = ['Below Q1', 'Q1-Median', 'Median-Q3', 'Above Q3']

		# Track min and max values
		current_min = min(counts)
		current_max = max(counts)
		if current_min < min_value:
			min_value = current_min
		if current_max > max_value:
			max_value = current_max

		# Plotting the bars for each feature at the offset position
		ax.bar([p + bar_width * i for p in positions], counts, bar_width, color=colors[i % len(colors)], edgecolor='black', label=feature, alpha=0.7)

	ax.set_title('Quartile Distribution for Selected Features')
	ax.set_ylabel('Count')
	ax.set_xlabel('Quartile Category')
	ax.set_xticks([p + bar_width * (3) / 2 for p in positions])
	ax.set_xticklabels(labels)
	ax.legend(loc='upper right')
	ax.grid(True)

	# Add annotations for min and max values
	ax.annotate(f'Min: {min_value}', xy=(0.5, 0.95), xycoords='axes fraction', ha='center', fontsize=10, color='black')
	ax.annotate(f'Max: {max_value}', xy=(0.5, 0.90), xycoords='axes fraction', ha='center', fontsize=10, color='black')
	ax.set_ylim(min_value - 1, max_value + 1)
	plt.draw()

	def on_key(event):
		if event.key == 'escape':
			plt.close(fig)
	fig.canvas.mpl_connect('key_press_event', on_key)

	plt.show()

if __name__ == "__main__":
	try:
		main()
	except Exception as error:
		print(error)
