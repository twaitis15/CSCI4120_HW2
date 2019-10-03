from sklearn.cluster import KMeans
from scipy.stats import mode
from sklearn.datasets import make_blobs
from yellowbrick.cluster import KElbowVisualizer
from sklearn.datasets.samples_generator import make_blobs
import sklearn.metrics as mtr 
from sklearn.metrics import accuracy_score
import numpy as np 
import pandas as pd
x, y_true = make_blobs(n_samples=300, centers=4,
                       cluster_std=0.60, random_state=0)


# TODO determine the best k for k-means
# the best k is 3
model = KMeans()
model.fit(x)
labels = model.predict(x)
visualView = KElbowVisualizer(model, k=(1,8)) 
visualView.fit(x)
visualView.poof()     # the plots is showing us a line on k=3

# TODO calculate accuracy for best
accuracy_score(y_true, labels)
print(accuracy_score)

# TODO draw a confusion matrix
dataFrame = pd.DataFrame(x, columns=['feature_A','feature_B'])
clusterer = KMeans(4, random_state=0)
clusterer.fit(x)

dataFrame['group'] = clusterer.predict(x)

dataFrame.head(5)

