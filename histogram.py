import pandas as pd
import matplotlib.pyplot as plt
import describe as desc
import math

def log_std(data, Feature):
	std_deviation = desc.std(data, Feature)
	if std_deviation <= 0:
		return float('nan')
	return math.log(std_deviation)

def main():
	data = pd.read_csv('datasets/dataset_train.csv')

	numeric_columns = data.select_dtypes(include=[float, int]).columns[1:]
	log_std_devs = {col: log_std(data, col) for col in numeric_columns}

	lowest_log_std_col = min(log_std_devs, key=log_std_devs.get)
	house_colors = {'Gryffindor': 'crimson', 'Hufflepuff': 'gold', 'Ravenclaw': 'royalblue', 'Slytherin': 'forestgreen'}
	house_data = data[['Hogwarts House', lowest_log_std_col]].dropna()
	houses = house_data['Hogwarts House'].unique()
	house_values = {house: house_data[house_data['Hogwarts House'] == house][lowest_log_std_col].tolist() for house in houses}

	min_val = house_data[lowest_log_std_col].min()
	max_val = house_data[lowest_log_std_col].max()

	fig, ax = plt.subplots(figsize=(12, 6))

	def display_plot(view):
		ax.clear()
		if view == 1:
			ax.bar(log_std_devs.keys(), log_std_devs.values())
			ax.set_ylabel('Logarithm of Standard Deviation')
			ax.set_title('Logarithm of Standard Deviation of Each Numeric Column (Skipping First Two Columns)')
			plt.xticks(rotation=45, ha='right')
			ax.set_ylim(-0.05, max(log_std_devs.values()) * 1.1)
		else:
			for house in houses:
				ax.hist(house_values[house], bins=20, range=(min_val, max_val), alpha=0.5, edgecolor='black', label=house,
						color=house_colors.get(house, 'gray'), density=True, stacked=True)

			ax.set_xlabel(lowest_log_std_col, fontsize=12)
			ax.set_ylabel('Density', fontsize=12)
			ax.set_title(f'Distribution of {lowest_log_std_col} by Hogwarts House', fontsize=16)
			ax.legend(title='Hogwarts House', fontsize=10)
			ax.set_ylim(0, ax.get_ylim()[1])
		plt.tight_layout()

	view = 1
	display_plot(view)

	def on_key(event):
		nonlocal view
		if event.key == ' ':
			view *= -1
			display_plot(view)
			plt.draw()
		elif event.key == 'escape':
			plt.close()

	fig.canvas.mpl_connect('key_press_event', on_key)
	
	plt.tight_layout()
	plt.show()

if __name__ == "__main__":
	try:
		main()
	except Exception as error:
		print(error)
