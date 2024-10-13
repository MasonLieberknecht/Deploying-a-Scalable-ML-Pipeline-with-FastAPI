# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
This model is a Random Forest Classifier trained to predict whether a person earns more than $50,000 per year. The data used to train the model comes from the 1994 Census dataset. Random Forest is a type of model that uses many decision trees to make a final prediction.

## Intended Use
The model is intended to help predict whether a person earns more than $50,000 based on features like their occupation, education, and marital status.
## Training Data
The model was trained using a dataset from the 1994 U.S. Census, which contains information about individuals such as their age, education, work status, and other personal characteristics. The dataset was split into training data to make model predictions and test data to see how well the model actually works.
## Evaluation Data
The test data, which was 20% of the original dataset, was used to evaluate the model.
## Metrics
Precision to measure how often the model if correct. THe models prediction is .07419 or 74%. Meaning the model was correct 74% of the time predicting people making over $50,000. Recall measures how many high earners the model identified correctly. The recall was 0.6384 or 64%. Meaning the model identified the high earners 64% of the time. F1 score is a combination of the other two metrics to give a balanced meaure of performace. THe F1 score is 0.6863

## Ethical Considerations
Ethical considerations of the data could be underrepresentation of certain groups of people leading to bias in the data. THe data also replies on personal data, which could be seen as an invasion of privacy or lead certain people to not want their data included. 

## Caveats and Recommendations
This model shouldn't be used to make important decisions, like those about money or jobs. It was trained on data that's almost 30 years old, so it might not work well with more recent information. Also, the model might not do a good job for certain groups of people who weren't well represented in the original data. This model is mostly meant for learning and practice, not for real-life use. If someone wanted to use it for real decisions, they would need more up-to-date and complete data. It's also important to make sure models like this are fair and don't treat some people unfairly, especially if they're being used in important areas like hiring or finances.