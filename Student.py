import numpy as np
from sklearn.cluster import KMeans
import pandas
import array
#data
data = np.array([
    [9.2, 95, 92, 90, 95, 92],
    [8.4, 88, 86, 85, 90, 88],
    [8.6, 90, 85, 88, 92, 86],
    [9.8, 98, 95, 97, 99, 96],
    [9.5, 95, 94, 93, 97, 95],
    [8.3, 85, 80, 82, 88, 80],
    [7.8, 80, 75, 78, 85, 75],
    [7.9, 82, 78, 80, 88, 80],
    [8, 85, 80, 82, 90, 78],
    [9.2, 92, 90, 88, 94, 90],
    [8.3, 85, 82, 80, 90, 80],
    [8.4, 86, 84, 82, 92, 82],
    [8.56, 88, 86, 84, 94, 84],
    [8.96, 92, 90, 88, 96, 88],
    [7.45, 75, 78, 70, 80, 70],
    [8.34, 85, 80, 82, 88, 80],
    [8.56, 88, 86, 84, 92, 84],
    [7.95, 80, 82, 78, 85, 78],
    [6.98, 70, 72, 68, 75, 70],
    [8.95, 90, 88, 86, 92, 86],
    [7.98, 80, 82, 78, 85, 78],
    [8.96, 92, 90, 88, 96, 88],
    [9.57, 95, 94, 92, 98, 92],
    [9.64, 95, 96, 94, 98, 94],
    [6.2, 71, 68, 82, 74, 60],
    [7.86, 78, 81, 80, 84, 80],
    [8.96, 90, 84, 88, 90, 85],
    [9.8, 95, 93, 97, 95, 98],
    [9, 90, 89, 88, 91, 83],
    [9.03, 92, 90, 82, 94, 96],
    [8.61, 90, 87, 88, 95, 86],
    [7.45, 70, 84, 82, 78, 80],
    [6.85, 79, 70, 70, 72, 69],
    [9.26, 94, 85, 88, 90, 86],
    [9.31, 96, 85, 90, 90, 85],
    [8.45, 90, 86, 85, 90, 82],
    [7.46, 86, 70, 82, 81, 77],
    [7.37, 80, 76, 78, 84, 79],
    [8.04, 82, 92, 75, 80, 85],
    [8.36, 90, 83, 76, 77, 96],
    [9.6, 94, 96, 95, 94, 93],
    [9.4, 92, 93, 88, 91, 90],
    [9.21, 90, 91, 88, 88, 85],
    [8.44, 84, 77, 80, 86, 90],
    [9.61, 96, 93, 90, 95, 92],
    [9.86, 98, 92, 95, 96, 97],
    [8.5, 92, 84, 77, 75, 94],
    [9.6, 95, 90, 89, 94, 91],
    [9.77, 90, 94, 89, 87, 95],
    [8.54, 86, 78, 84, 88, 92],
    [6.78, 70, 72, 68, 75, 70],
    [9.32, 92, 90, 88, 92, 90],
    [9.64, 95, 94, 92, 98, 94],
    [8.7, 85, 82, 88, 88, 82],
    [8.5, 82, 80, 82, 85, 81],
    [9.06, 90, 88, 90, 92, 87],
    [7.63, 78, 75, 80, 80, 75],
    [9.87, 95, 94, 96, 98, 94],
    [8.63, 85, 88, 82, 87, 82],
    [9.69, 94, 96, 93, 97, 94],
    [8.65, 87, 78, 88, 83, 84],
    [5.8, 55, 62, 60, 55, 65],
    [9.55,92, 93, 95, 95, 87],
    [9.24, 88, 90, 97, 90, 87],
    [8.76, 87, 89, 84, 88, 89],
    [7.86, 78, 80, 78, 82, 81],
    [9.54, 93, 90, 96, 95, 86],
    [8.64, 89, 82, 83, 89, 88],
    [9.08, 92, 88, 92, 90, 86],
    [7.98, 74, 78, 81, 91, 70],
    [8.94, 90, 88, 83, 88, 89],
    [8.54, 84, 83, 89, 90, 88],
    [9.76, 95, 90, 97, 92, 95],
    [9.45, 92, 95, 92, 90, 89],
    [8.75, 89, 90, 84, 82, 87],
    [6.98, 70, 69, 75, 80, 78],
    [7.95, 71, 81, 75, 89, 87],
    [8.76, 84, 89, 87, 89, 88],
    [9.56, 91, 96, 95, 88, 92],
    [9.45, 88, 93, 90, 89, 92],
    [9.57, 94, 90, 93, 86, 95],
    [6.4, 75, 65, 70, 71, 78],
    [8.56, 86, 87, 90, 82, 88],
    [8.43, 84, 83, 89, 90, 88],
    [8.52, 82, 21, 89, 88, 90],
    [8.37, 80, 81, 87, 84, 86],
    [8.54, 85, 87, 82, 90, 89],
    [8.23, 77, 82, 90, 87, 91],
    [8.62, 87, 92, 88, 82, 87],
    [9.76, 97, 92, 94, 96, 90],
    [9.47, 93, 84, 88, 95, 89],
    [9.24, 90, 88, 96, 89, 89],
    [9.64, 91, 95, 90, 94, 95],
    [9.31, 91, 88, 85, 98, 96],
    [8.9, 89, 89, 91, 85, 87],
    [5.86, 60, 61, 58, 65, 60],
    [9.05, 93, 92, 88, 90, 86],
    [9.45, 87, 84, 93, 89, 95],
    [8.33, 85, 84, 87, 81, 86],
    [9.22, 92, 86, 95, 89, 88],
    [9.83, 96, 95, 97, 92, 96],
    [8.95, 91, 83, 89, 88, 82],
    [8.99, 86, 95, 89, 83, 88],
    [8.64, 89, 88, 92, 84, 85],
    [7.97, 69, 83, 78, 89, 87],
    [9.74, 95, 92, 91, 91, 93],
    [8.54, 88, 85, 90, 81, 86],
    [9.74, 95, 92, 91, 91, 93],
    [8.73, 88, 86, 88, 82, 87],
    [9.53, 96, 84, 87, 90, 95],
    [6.56, 68, 69, 73, 81, 76],
    [8.68, 89, 90, 88, 86, 85],
    [8.09, 83, 82, 86, 78, 88],
    [9.04, 87, 91, 92, 86, 90],
    [8.8, 90, 87, 89, 89, 85],
    [9.5, 87, 84, 93, 89, 95],
    [9.4, 83, 86, 89, 92, 94],
    [8.77, 86, 88, 90, 86, 85],
    [9.72, 90, 92, 95, 89, 93],
    [9, 88, 89, 95, 90, 91],
    [8,99,80,94,89,88]
    
    
])
# Convert data to numpy array
X = np.array(data).reshape(-1, 1)
# Perform k-means clustering
kmeans = KMeans(n_clusters=5, random_state=42)
kmeans.fit(data)

# Get cluster labels
labels = kmeans.labels_
centers = kmeans.cluster_centers_

# Print cluster labels for each student
for i, label in enumerate(labels):
    if label==0:
        print(f"Student {i+1}: CGPA {data[i][0]}, Quiz {data[i][1]},\
        Assignment {data[i][2]},Midtest {data[i][3]},\
        Lab Performance {data[i][4]} -> Cluster {label+1}",\
              "; Performance of student in next semester : Excellent")
    if label==1:
        print(f"Student {i+1}: CGPA {data[i][0]}, Quiz {data[i][1]},\
        Assignment {data[i][2]},Midtest {data[i][3]},\
        Lab Performance {data[i][4]} -> Cluster {label+1}",\
              "; Performance of student in next semester : Very Poor")
    if label==2:
        print(f"Student {i+1}: CGPA {data[i][0]}, Quiz {data[i][1]},\
        Assignment {data[i][2]},Midtest {data[i][3]},\
        Lab Performance {data[i][4]} -> Cluster {label+1}",\
              "; Performance of student in next semester : Average")
    if label==3:
        print(f"Student {i+1}: CGPA {data[i][0]}, Quiz {data[i][1]},\
        Assignment {data[i][2]},Midtest {data[i][3]},\
        Lab Performance {data[i][4]} -> Cluster {label+1}",\
              "; Performance of student in next semester : Good")
    if label==4:
        print(f"Student {i+1}: CGPA {data[i][0]}, Quiz {data[i][1]},\
        Assignment {data[i][2]},Midtest {data[i][3]},\
        Lab Performance {data[i][4]} -> Cluster {label+1}",\
              "; Performance of student in next semester : Poor")
    # Print cluster centers
print("\nCluster Centers:")
for i, center in enumerate(centers):
    if i==0:
        print(f"Cluster {i+1}: Center {center[0]}" ,\
              "; Performance of student in next semester : Excellent")
    if i==3:
        print(f"Cluster {i+1}: Center {center[0]}" ,\
              "; Performance of student in next semester : Good")
    if i==2:
        print(f"Cluster {i+1}: Center {center[0]}" ,\
              "; Performance of student in next semester : Average")
    if i==4:
        print(f"Cluster {i+1}: Center {center[0]}" ,\
              "; Performance of student in next semester : Poor")
    if i==1:
        print(f"Cluster {i+1}: Center {center[0]}" ,\
              "; Performance of student in next semester : Very poor")
