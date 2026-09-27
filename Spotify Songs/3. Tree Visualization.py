#tree visualization
import matplotlib.pyplot as plt
from sklearn import tree
plt.figure(figsize=(25,15))
tree.plot_tree(dt,feature_names=x.columns,
              class_names=df['mode'].astype(str).unique(),
              filled=True)
plt.show()
