# Decision-Tree
Decision Trees (DTs) are a non-parametric supervised learning method used for classification and regression. The goal is to create a model that predicts the value of a target variable by learning simple decision rules inferred from the data features. A tree can be seen as a piecewise constant approximation.

For instance, in the example below, decision trees learn from data to approximate a sine curve with a set of if-then-else decision rules. The deeper the tree, the more complex the decision rules and the fitter the model.

![image](https://user-images.githubusercontent.com/109084435/201509630-cd6d90bc-a661-4f01-a39a-2ec41e43d189.png)

#### Advantages :

1.Simple to understand and to interpret. Trees can be visualized.

2.Requires little data preparation. Other techniques often require data normalization, dummy variables need to be created and blank values to be removed. Note however that this module does not support missing values.

3.The cost of using the tree (i.e., predicting data) is logarithmic in the number of data points used to train the tree.

4.Able to handle both numerical and categorical data. However, the scikit-learn implementation does not support categorical variables for now. Other techniques are usually specialized in analyzing datasets that have only one type of variable. See algorithms for more information.

5.Able to handle multi-output problems.

6.Uses a white box model. If a given situation is observable in a model, the explanation for the condition is easily explained by boolean logic. By contrast, in a black box model (e.g., in an artificial neural network), results may be more difficult to interpret.

7.Possible to validate a model using statistical tests. That makes it possible to account for the reliability of the model.

8.Performs well even if its assumptions are somewhat violated by the true model from which the data were generated.

#### Disadvantages :

1.Decision-tree learners can create over-complex trees that do not generalize the data well. This is called overfitting. Mechanisms such as pruning, setting the minimum number of samples required at a leaf node or setting the maximum depth of the tree are necessary to avoid this problem.

2.Decision trees can be unstable because small variations in the data might result in a completely different tree being generated. This problem is mitigated by using decision trees within an ensemble.

3.Predictions of decision trees are neither smooth nor continuous, but piecewise constant approximations as seen in the above figure. Therefore, they are not good at extrapolation.

4.The problem of learning an optimal decision tree is known to be NP-complete under several aspects of optimality and even for simple concepts. Consequently, practical decision-tree learning algorithms are based on heuristic algorithms such as the greedy algorithm where locally optimal decisions are made at each node. Such algorithms cannot guarantee to return the globally optimal decision tree. This can be mitigated by training multiple trees in an ensemble learner, where the features and samples are randomly sampled with replacement.

5.There are concepts that are hard to learn because decision trees do not express them easily, such as XOR, parity or multiplexer problems.

6.Decision tree learners create biased trees if some classes dominate. It is therefore recommended to balance the dataset prior to fitting with the decision tree.

#### Decision Tree Terminologies

1.Root Node: Root node is from where the decision tree starts. It represents the entire dataset, which further gets divided into two or more homogeneous sets.

2.Leaf Node: Leaf nodes are the final output node, and the tree cannot be segregated further after getting a leaf node.

3.Splitting: Splitting is the process of dividing the decision node/root node into sub-nodes according to the given conditions.

4.Branch/Sub Tree: A tree formed by splitting the tree.

5.Pruning: Pruning is the process of removing the unwanted branches from the tree.

6.Parent/Child node: The root node of the tree is called the parent node, and other nodes are called the child nodes.

![image](https://user-images.githubusercontent.com/109084435/201510998-13b1dabd-d5ef-492c-b4cb-f615a17e05cb.png)

#### How does the Decision Tree algorithm Work?

In a decision tree, for predicting the class of the given dataset, the algorithm starts from the root node of the tree. This algorithm compares the values of root attribute with the record (real dataset) attribute and, based on the comparison, follows the branch and jumps to the next node.

For the next node, the algorithm again compares the attribute value with the other sub-nodes and move further. It continues the process until it reaches the leaf node of the tree.

##### Step-1: 

Begin the tree with the root node, says S, which contains the complete dataset.

##### Step-2: 

Find the best attribute in the dataset using Attribute Selection Measure (ASM).

##### Step-3: 

Divide the S into subsets that contains possible values for the best attributes.

##### Step-4:

Generate the decision tree node, which contains the best attribute.

##### Step-5:

Recursively make new decision trees using the subsets of the dataset created in step -3. Continue this process until a stage is reached where you cannot further classify the nodes and called the final node as a leaf node.

### Example:

Suppose there is a candidate who has a job offer and wants to decide whether he should accept the offer or Not. So, to solve this problem, the decision tree starts with the root node (Salary attribute by ASM). The root node splits further into the next decision node (distance from the office) and one leaf node based on the corresponding labels. The next decision node further gets split into one decision node (Cab facility) and one leaf node. Finally, the decision node splits into two leaf nodes (Accepted offers and Declined offer). Consider the below diagram:

![image](https://user-images.githubusercontent.com/109084435/201509943-7489e2e9-906d-41ce-bd75-fc2508fcb384.png)

#### Attribute Selection Measures

While implementing a Decision tree, the main issue arises that how to select the best attribute for the root node and for sub-nodes. So, to solve such problems there is a technique which is called as Attribute selection measure or ASM. By this measurement, we can easily select the best attribute for the nodes of the tree. There are popular techniques for ASM, which are:

##### 1.Information Gain
##### 2.Gini Index
##### 3.Entropy
##### 4.Chi-Square

#### 1. Information Gain:

-Information gain is the measurement of changes in entropy after the segmentation of a dataset based on an attribute.

-It calculates how much information a feature provides us about a class.

-According to the value of information gain, we split the node and build the decision tree.

-A decision tree algorithm always tries to maximize the value of information gain, and a node/attribute having the highest information gain is split first. It can be calculated using the below formula:

##### Information Gain= Entropy(S)- [(Weighted Avg) *Entropy(each feature)  

#### 2. Gini Index:

-Gini index is a measure of impurity or purity used while creating a decision tree in the CART(Classification and Regression Tree) algorithm.

-An attribute with the low Gini index should be preferred as compared to the high Gini index.

-It only creates binary splits, and the CART algorithm uses the Gini index to create binary splits.

-Gini index can be calculated using the below formula:

##### Gini Index= 1- ∑jPj*2

#### 3.Entropy

Entropy is a measure of the randomness in the information being processed.It is a metric to measure the impurity in a given attribute. The higher the entropy, the harder it is to draw any conclusions from that information. Flipping a coin is an example of an action that provides information that is random.

##### Entropy(s)= -P(yes)log2 P(yes)- P(no) log2 P(no)

#### 4.Chi-Square

The acronym CHAID stands for Chi-squared Automatic Interaction Detector. It is one of the oldest tree classification methods. It finds out the statistical significance between the differences between sub-nodes and parent node. We measure it by the sum of squares of standardized differences between observed and expected frequencies of the target variable.

It works with the categorical target variable “Success” or “Failure”. It can perform two or more splits. Higher the value of Chi-Square higher the statistical significance of differences between sub-node and Parent node.

It generates a tree called CHAID (Chi-square Automatic Interaction Detector).

Mathematically, Chi-squared is represented as:

![image](https://user-images.githubusercontent.com/109084435/201510670-56350dc0-dbc3-4967-adfb-330412fbdd97.png)

### How to avoid/counter Overfitting in Decision Trees?

The common problem with Decision trees, especially having a table full of columns, they fit a lot. Sometimes it looks like the tree memorized the training data set. If there is no limit set on a decision tree, it will give you 100% accuracy on the training data set because in the worse case it will end up making 1 leaf for each observation. Thus this affects the accuracy when predicting samples that are not part of the training set.

Here are two ways to remove overfitting:

1.Pruning Decision Trees.

2.Random Forest.

### 1.Pruning Decision Trees

The splitting process results in fully grown trees until the stopping criteria are reached. But, the fully grown tree is likely to overfit the data, leading to poor accuracy on unseen data.

In pruning, you trim off the branches of the tree, i.e., remove the decision nodes starting from the leaf node such that the overall accuracy is not disturbed. This is done by segregating the actual training set into two sets: training data set, D and validation data set, V. Prepare the decision tree using the segregated training data set, D. Then continue trimming the tree accordingly to optimize the accuracy of the validation data set, V.

![image](https://user-images.githubusercontent.com/109084435/201510776-9d76e966-ebab-4d4b-874b-bb7e992dadf9.png)

In the above diagram, the ‘Age’ attribute in the left-hand side of the tree has been pruned as it has more importance on the right-hand side of the tree, hence removing overfitting.

### 2.Random Forest

Random Forest is an example of ensemble learning, in which we combine multiple machine learning algorithms to obtain better predictive performance.

##### Why the name “Random”?

Two key concepts that give it the name random:

1.A random sampling of training data set when building trees.

2.Random subsets of features considered when splitting nodes.

A technique known as bagging is used to create an ensemble of trees where multiple training sets are generated with replacement.

In the bagging technique, a data set is divided into N samples using randomized sampling. Then, using a single learning algorithm a model is built on all samples. Later, the resultant predictions are combined using voting or averaging in parallel. 









