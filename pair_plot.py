import pandas as pd
import matplotlib.pyplot as plt
import math


def log_std(data, feature):
    std_deviation = data[feature].std()
    return math.log(std_deviation) if std_deviation > 0 else float("nan")


def normalize_column(data, column):
    min_val = data[column].min()
    max_val = data[column].max()
    return (data[column] - min_val) / (max_val - min_val)


def set_axis_labels(axes, courses):
    for j, course in enumerate(courses):
        ax_bottom = axes[-1, j]
        ax_bottom.set_xlabel(course, fontsize=8, rotation=45, ha="right")
        ax_bottom.xaxis.set_label_position("bottom")
        ax_bottom.xaxis.set_label_coords(0.5, -0.35)

    for i, course in enumerate(courses):
        ax_left = axes[i, 0]
        ax_left.set_ylabel(course, fontsize=8, rotation=0, ha="right", va="center")
        ax_left.yaxis.set_label_position("left")
        ax_left.yaxis.set_label_coords(-0.35, 0.5)


data = pd.read_csv("datasets/dataset_train.csv")

numeric_columns = data.select_dtypes(include=[float, int]).columns[1:]
log_std_devs = {col: log_std(data, col) for col in numeric_columns}
lowest_log_std_col = min(log_std_devs, key=log_std_devs.get)

house_colors = {
    "Gryffindor": "crimson",
    "Hufflepuff": "gold",
    "Ravenclaw": "royalblue",
    "Slytherin": "forestgreen",
}

courses = data.select_dtypes(include=[float, int]).columns[1:]
normalized_data = data.copy()
for course in courses:
    normalized_data[course] = normalize_column(data, course)

colors = normalized_data["Hogwarts House"].map(house_colors)
number_of_courses = len(courses)
fig_width = 4 * number_of_courses
fig_height = 2 * number_of_courses

fig, axes = plt.subplots(
    number_of_courses,
    number_of_courses,
    figsize=(fig_width, fig_height),
    sharex=True,
    sharey=True,
    constrained_layout=True,
)

for i in range(number_of_courses):
    for j in range(number_of_courses):
        ax = axes[i, j]
        if i != j:
            ax.scatter(
                normalized_data[courses[j]],
                normalized_data[courses[i]],
                c=colors,
                s=5,
                alpha=0.5,
                edgecolors="w",
                linewidth=0.5,
            )
            ax.set_aspect("auto")
        else:
            ax.axis("off")
        ax.tick_params(axis="both", which="major", labelsize=6)

for i in range(number_of_courses - 1):
    for j in range(1, number_of_courses):
        axes[i, j].set_xticklabels([])
        axes[i, j].set_yticklabels([])

set_axis_labels(axes, courses)
plt.show()
