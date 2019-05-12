from pathlib import Path
from time import time

import numpy as np
import umap
from sklearn.datasets import load_digits, fetch_mldata, fetch_openml
import matplotlib.pyplot as plt
import seaborn as sns

k = 15
min_dist = 0.1
metric = 'euclidean'
full = True

if full:
    #mnist = fetch_mldata("MNIST original")
    mnist = fetch_openml('mnist_784', version=1)
    target = np.array([float(v) for v in mnist.target])
else:
    mnist = load_digits()
    target = mnist.target

#reducer = umap.UMAP(random_state=42, )
#embedding = reducer.fit_transform(mnist.data)

reducer = umap.UMAP(random_state=42,
                    n_neighbors=k,
                    min_dist=min_dist,
                    metric=metric)

t = time()
embedding = reducer.fit_transform(mnist.data)
t = time() - t


sns.set(context="paper", style="white")
fig, ax = plt.subplots(figsize=(12, 10))
plt.scatter(embedding[:, 0], embedding[:, 1], c=target/max(target), cmap="Spectral") # s = 6 ** 2 by default
plt.setp(ax, xticks=[], yticks=[])
plt.title("MNIST data embedded into two dimensions by UMAP", fontsize=18)

# export = Path('export')
# path = export / f"example-{'full' if full else 'small'}-{k}-{min_dist}-{metric}.png"
# plt.savefig(path, bbox_inches='tight')

plt.show()
print(f'UMAP takes {t:.3f}s to finish')
