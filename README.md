# Decision-Tree-Classifier

* Description
  * Custom decision tree classifier implemented in C++.
  * Uses a linked-list-based general tree structure (TreeNode and EdgeNode).
  * Builds classification trees from tabular datasets using entropy and information gain (ID3-style learning).
  * Supports datasets with any number of categorical features.

## Features
  * Automatic decision tree construction from training data.
  * Entropy and information gain calculations for feature selection.
  * Recursive classification through tree traversal.
  * Node count and leaf node count utilities.
  * Majority-class fallback when no further information gain is possible.
  * Recursive memory management with destructor-based cleanup.

## Concepts Demonstrated
  * Tree data structures
  * Linked lists
  * Recursion
  * Dynamic memory allocation
  * Entropy and information gain
  * Basic machine learning / decision tree learning

## Testing
  * Standard multi-feature classification datasets.
  * Single-feature datasets.
  * Uniform outcome datasets.
  * Majority-class fallback cases.
  * Datasets with unusual feature names and values.

## Author

Nathaniel Darren Lim
