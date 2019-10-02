from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from yellowbrick.cluster import KElbowVisualizer
from sklearn.datasets.samples_generator import make_blobs
import sklearn.metrics as mtr 

x, y_true = make_blobs(n_samples=300, centers=4,
                       cluster_std=0.60, random_state=0)


# TODO determine the best k for k-means
# the best k is 3
model = KMeans()
visualView = KElbowVisualizer(model, k=(1,8)) 
visualView.fit(x)
visualView.poof()     # the plots is showing us a line on k=3

# TODO calculate accuracy for best K
# TODO draw a confusion matrix
