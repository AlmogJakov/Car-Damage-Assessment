
<h1>Car Damage Assessment</h1>

Classification of car damage images by a variety of machine learning algorithms    
     
In addition:
- Examining the accuracy of each algorithm
- Examining the effect of image properties on learning (we will examine this by using a variety of image processing algorithms to perform "minipulations" on the images)

------------------------------

![image](https://user-images.githubusercontent.com/68508896/178482762-3aae3c24-9edd-4a58-97ee-56214d70626b.png)


<h2>Database:</h2>

The car damage dataset contains approximately 1,500 unique RGB images with the dimensions 224 x 224 pixels, and is split into a training- and a validation subset.

Classes   
The illustration above show sample images from the various classes in the dataset. Note that the unknown class contains images of cars that are in either pristine or wrecked condition.

Each collected image represents one car with one specific type of damage.

Example images from each class; Broken headlamp, Broken tail lamp, Glass shatter, Door scratch, Door dent, Bumper dent, Bumper scratch, Unknown   

[Database Source](https://www.kaggle.com/datasets/hamzamanssor/car-damage-assessment)

<h2>Algorithms:</h2>

* <b>CNN</b> is a concept of a neural network, Its main attributes may be that it consists of convolution layers, pooling layers , activation layers etc.
* <b>VGG</b> is a specific convolutional network designed for classification and localization.Like many other popular networks like Google-Net, Alex Net etc.
* <b>SVM</b> works by mapping data to a high-dimensional feature space so that data points can be categorized, even when the data are not otherwise linearly separable. A separator between the categories is found, then the data are transformed in such a way that the separator could be drawn as a hyperplane. Following this, characteristics of new data can be used to predict the group to which a new record should belong [(Source)](https://www.ibm.com/docs/it/spss-modeler/SaaS?topic=models-how-svm-works).
* <b>KNN</b> works by finding the distances between a query and all the examples in the data, selecting the specified number examples (K) closest to the query, then votes for the most frequent label (in the case of classification) or averages the labels (in the case of regression) [(Source)](https://towardsdatascience.com/machine-learning-basics-with-the-k-nearest-neighbors-algorithm-6a6e71d01761).
* <b>Decision trees</b> use multiple algorithms to decide to split a node into two or more sub-nodes. The creation of sub-nodes increases the homogeneity of resultant sub-nodes. In other words, we can say that the purity of the node increases with respect to the target variable. The decision tree splits the nodes on all available variables and then selects the split which results in most homogeneous sub-nodes [(Source)](https://www.kdnuggets.com/2020/01/decision-tree-algorithm-explained.html).


<h2>Algorithms Results:</h2>

![image](https://user-images.githubusercontent.com/68508896/178959234-9cdcb88d-9256-493c-992f-9162dd7ba0d9.png)


<h2>Database Tweaks Comparison Results:</h2>  

- Examining the effect of image properties on learning    
- Comparison using the Support Vector Machine algorithm on VGG vectors
- Comparing results with success rates
- Mark: HLT=Hough Line Transform

![image](https://user-images.githubusercontent.com/68508896/178497518-a0811d4c-3ae5-4c93-87ad-ab43f76bec32.png)


--------   

Itay Github: [https://github.com/itay-rafee](https://github.com/itay-rafee)  
Almog Github: [https://github.com/AlmogJakov](https://github.com/AlmogJakov)
