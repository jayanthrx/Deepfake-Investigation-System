---
license: apache-2.0
metrics:
- accuracy
- f1
base_model:
- google/vit-base-patch16-224-in21k
---
Checks whether an image is real or fake (AI-generated).

**Note to users who want to use this model in production**

Beware that this model is trained on a dataset collected about 3 years ago. 
Since then, there is a remarkable progress in generating deepfake images with common AI tools, resulting in a significant concept drift. 
To mitigate that, I urge you to retrain the model using the latest available labeled data. 
As a quick-fix approach, simple reducing the threshold (say from default 0.5 to 0.1 or even 0.01) of labelling image as a fake may suffice. 
However, you will do that at your own risk, and retraining the model is the better way of handling the concept drift.

See https://www.kaggle.com/code/dima806/deepfake-vs-real-faces-detection-vit for more details.

```
Classification report:

              precision    recall  f1-score   support

        Real     0.9921    0.9933    0.9927     38080
        Fake     0.9933    0.9921    0.9927     38081

    accuracy                         0.9927     76161
   macro avg     0.9927    0.9927    0.9927     76161
weighted avg     0.9927    0.9927    0.9927     76161
```