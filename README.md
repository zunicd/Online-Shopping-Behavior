<center><h1>Online Shopping Behavior</h1></center>



Online shopping behavior is the process by which consumers search for, select, purchase, use, and dispose of goods and services, over the internet. For the ecommerce platform, one of the most important questions is, whether the customer is just browsing or actually buying. 

Customers are very heterogenous so it is important that the sellers do not treat them in the same way. They always want to leverage their resources to find and keep the customers in which they have confidence that they can more likely to purchase. The sellers could take some proactive action, like time-limited coupons or free trials, to push customers to purchase. By targeting the right customers the sellers could improve retention and increase sales and profits.

In this project we did an intensive analysis of consumer behavioral and performing the following tasks:

1. Classification
   - Exploratory Data Analysis (EDA)

   - Modelling

   - Explainability with SHAP

2. Customer Segmentation
   - KMeans Clustering

3. Semi-Supervised Learning
   - Label Spreading

The complete code can be found in the notebook *online_shopping_behavior.ipynb* and the Python script *metrics_utilities.py* which contents two helping functions..

# Data

The input dataset  is *dat/online_shoppers_intention.csv.gz*, and its source is [Online Shoppers Purchasing Intention Dataset](https://archive.ics.uci.edu/ml/datasets/Online+Shoppers+Purchasing+Intention+Dataset).

The dataset was collected from an online bookstore. It consists of feature vectors belonging to 12,330 sessions.
It was formed so that each session would belong to a different user in a 1-year period to avoid any tendency to a specific campaign, special day, user profile, or period.

- The dataset consists of 10 numerical and 8 categorical variables.
- Variable `Revenue` is the target label. This binary variable is imbalanced, specifically 84.5% of user journeys did NOT result in a purchase; in other words `Revenue` is False.

# 1. Classification

We built a predictive classification model using data entries corresponding to the months of June—December as training set, and those corresponding to February—March as test set. 

We fit logistic regression and random forest classification models and found and explained important features. 

We performed a hyper-parameter fitting process and displayed classification metrics.

#  2. Customer Segmentation

We generated user-behavior clusters based on the purchasing behavior data for the complete dataset. 

We performed a detailed analysis of the KMeans clusters and plotted cluster images generated for the data.

The most interesting behavior is a pretty good separation between new shoppers and returniong shoppers for all cluster configurations (2, 3, 5, clusters)

# 3. Semi-Supervised Learning

We considered that we have training data with the `Revenue` attribute for records from June—September only. 

For all records from October—December, however, `Revenue` attribute is missing. We built a semi-supervised self-labeling model to estimate `Revenue` for the missing records in October—December and then fit the classifier. 

We reported classification performance on February—March data set with and without the self-labeled data.

