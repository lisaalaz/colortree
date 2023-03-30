# colortree
A simple Python library for building, pruning, evaluating and plotting colorful decision trees.

To install run `pip install git+https://github.com/LisaAlaz/colortree@main`

See example usage below.

![tree](https://user-images.githubusercontent.com/89645136/228981568-4c0a4fd9-0684-4044-89fa-24bc0ca00c93.png)

Example usage:

```
import colortree as ct
import numpy as np

train = np.loadtxt('sample_data/train.txt')
dev = np.loadtxt('sample_data/dev.txt')
test = np.loadtxt('sample_data/test.txt')

# Build and plot tree from train set
tree, depth = ct.make_tree(train)

# Build and plot optimally pruned tree from train set
tree, depth = ct.make_pruned_tree(train, dev, n_classes=4)

# Inference on one sample
pred = ct.predict_sample(tree, test[0])

# Inference on the test set
test_preds = ct.predict_test_set(tree, test)

# Print evaluation report
ct.eval_report(tree, test, n_classes=4)
```
