
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
The illustration abov show sample images from the various classes in the dataset. Note that the unknown class contains images of cars that are in either pristine or wrecked condition.

Each collected image represents one car with one specific type of damage.

Example images from each class; Broken headlamp, Broken tail lamp, Glass shatter, Door scratch, Door dent, Bumper dent, Bumper scratch, Unknown   

[Database Source](https://www.kaggle.com/datasets/hamzamanssor/car-damage-assessment)


<h2>Database tweaks comparison:</h2>  

- Examining the effect of image properties on learning    
- Comparison using the Support Vector Machine algorithm on VGG vectors
- Comparing results with success rates
- Mark: HLT=Hough Line Transform

![image](https://user-images.githubusercontent.com/68508896/178497518-a0811d4c-3ae5-4c93-87ad-ab43f76bec32.png)

