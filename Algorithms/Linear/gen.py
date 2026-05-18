# Generate a data for 1000 samples on a student "StudyHours" vs "GPA" dataset for Linear Regresssion test

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# Set random seed for reproducibility
np.random.seed(42)
# Generate synthetic data
num_samples = 1000
study_hours = np.random.uniform(0, 10, num_samples)  # Study hours
# Assume a linear relationship with some noise
gpa = 0.5 * study_hours + np.random.normal(0, 1, num_samples)  # GPA with noise
# Create a DataFrame
data = pd.DataFrame({'StudyHours': study_hours, 'GPA': gpa})
# Save to CSV           
data.to_csv('student_study_hours_gpa.csv', index=False)
