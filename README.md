# imgcls_utils
utilities and NN's required for building an image classifier 


flask_app.py - This file is used to load and deploy your ML model on a microWebService which accepts data in JSON format via
                the curl command <curl -X POST 0.0.0.0:5000/predict -H 'Content-Type: application/json'>


image_classifier_utils.py -This file contains defs for loading , processing , showing and predicting the class of the foriegn 
                          test images.

io.py - This file retrieves test image url from the MySql database (phpMyAdmin)  which is in the local machine 
        and predicts the class of the image passed as url and then uploads the image class name back to the MySql database.


train.py - This file is used to train your own image classifier based on AlexNet pretrained model .

predict.py - This file predicts the image class .

data_retriever_from_database.py  - This file fetches the image_url column value from the database running on localhost.


