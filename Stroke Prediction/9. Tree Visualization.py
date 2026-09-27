#TREE VISUALIZATION

import matplotlib.pyplot as plt
from sklearn import tree
plt.figure(figsize=(30,25))
tree.plot_tree(dt,
               feature_names=x.columns,
               class_names=df.stroke.astype(str).unique(),
               filled=True)
plt.show()
