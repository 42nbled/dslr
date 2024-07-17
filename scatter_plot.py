import pandas as pd
import matplotlib.pyplot as plt
import describe as desc
import math
from itertools import combinations


def log_std(data, Feature):
    std_deviation = desc.std(data, Feature)
    if std_deviation <= 0:
        return float("nan")
    return math.log(std_deviation)


data = pd.read_csv("datasets/dataset_train.csv")

numeric_columns = data.select_dtypes(include=[float, int]).columns[1:]
log_std_devs = {col: log_std(data, col) for col in numeric_columns}

lowest_log_std_col = min(log_std_devs, key=log_std_devs.get)
house_colors = {"Gryffindor": "crimson", "Hufflepuff": "gold", "Ravenclaw": "royalblue", "Slytherin": "forestgreen"}
house_data = data[["Hogwarts House", lowest_log_std_col]].dropna()
houses = house_data["Hogwarts House"].unique()
house_values = {
    house: house_data[house_data["Hogwarts House"] == house][lowest_log_std_col].tolist() for house in houses
}

# Selects columns with numeric values
courses = data.select_dtypes(include=[float, int]).columns[1:]

# Normalizes the grades using min-max normalization
normalized_data = data.copy()
for course in courses:
    min_grade = data[course].min()
    max_grade = data[course].max()
    normalized_data[course] = (data[course] - min_grade) / (max_grade - min_grade)

# Computes the correlation matrix
corr_matrix = normalized_data[courses].corr()

# Finds the pair with the highest absolute correlation
max_corr = 0
course_pair = (None, None)
for course1, course2 in combinations(courses, 2):
    current_corr = corr_matrix.at[course1, course2]
    if abs(current_corr) > max_corr:
        max_corr = abs(current_corr)
        course_pair = (course1, course2)

course1, course2 = course_pair
course_data = data[["Hogwarts House", course1, course2]].dropna()
colors = course_data["Hogwarts House"].map(house_colors)
plt.scatter(course_data[course1], course_data[course2], c=colors)
plt.xlabel(course1)
plt.ylabel(course2)
plt.title("Courses with the most similar grade distribution")
plt.show()
