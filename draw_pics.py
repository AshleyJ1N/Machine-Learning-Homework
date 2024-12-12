import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

gender_filename = f'sliding_window_f1_gender.csv'
age_filename = f'sliding_window_r2_age.csv'

Xtmp = pd.read_csv(gender_filename)
Xtmp = Xtmp.values
gender_raw = np.array(Xtmp)
Xtmp = pd.read_csv(age_filename)
Xtmp = Xtmp.values
age_raw = np.array(Xtmp)

gender_y_axis = [-1]*740
age_y_axis = [-1]*740
for x in range(0, 740):
    if x == 0:
        gender_y_axis[x] = gender_raw[0]
        age_y_axis[x] = age_raw[0]
    elif x == 1:
        gender_y_axis[x] = (gender_raw[0] + gender_raw[1])/2
        age_y_axis[x] = (age_raw[0] + age_raw[1]) / 2
    elif x == 738:
        gender_y_axis[x] = (gender_raw[736] + gender_raw[737])/2
        age_y_axis[x] = (age_raw[736] + age_raw[737]) / 2
    elif x == 739:
        gender_y_axis[x] = gender_raw[737]
        age_y_axis[x] = age_raw[737]
    else:
        gender_y_axis[x] = (gender_raw[x-2] + gender_raw[x-1] + gender_raw[x])/3
        age_y_axis[x] = (age_raw[x - 2] + age_raw[x - 1] + age_raw[x]) / 3

# gender折线图
x_axis = range(0, 740)
y_axis = gender_y_axis
plt.plot(x_axis, y_axis)
plt.savefig('new_sliding_window_f1_gender.png')

gender_y_axis = pd.DataFrame(gender_y_axis)
gender_y_axis.to_csv('new_sliding_window_f1_gender.csv')

# age折线图
x_axis = range(0, 740)
y_axis = age_y_axis
plt.plot(x_axis, y_axis)
plt.savefig('new_sliding_window_r2_age.png')

age_y_axis = pd.DataFrame(age_y_axis)
age_y_axis.to_csv('new_sliding_window_r2_age.csv')
