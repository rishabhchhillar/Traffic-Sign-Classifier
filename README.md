# Traffic Sign Recognition Program

Overview
---
In this project, I've used deep neural networks and convolutional neural networks to classify traffic signs. I trained and validated a model so it can classify traffic sign images using the [German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset). After the model was trained, I tried out the model on images of German traffic signs found on the web.

The Project
---
The goals / steps of this project are the following:
* Load the data set
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images

[//]: # (Image References)

[im01]: ./examples/train_set_counts.png "Train Set Counts"
[im02]: ./examples/test_set_counts.png "Test Set Counts"
[im03]: ./examples/valid_set_counts.png "Valid Set Counts"
[im04]: ./examples/grayscale.png "Grayscale Image"
[im05]: ./examples/normalized.png "Normalized Image"
[im06]: ./examples/1.png "German Traffic Sign"
[im07]: ./examples/2.png " German Traffic Sign"
[im08]: ./examples/3.png " German Traffic Sign"
[im09]: ./examples/4.png " German Traffic Sign"
[im10]: ./examples/5.png " German Traffic Sign"
[im11]: ./examples/6.png " German Traffic Sign"

### Data Set Summary & Exploration

* The size of training set is 34799.
* The size of the validation set is 4410.
* The size of test set is 12630.
* The shape of a traffic sign image is 32x32x1.
* The number of unique classes/labels in the data set is 43.

Here is an exploratory visualization of the data set. It is a bar chart showing the number of images in training, validation, and test sets.

![alt text][im01]
![alt text][im02]
![alt text][im03]

### Design and Test a Model Architecture

For pre-processing, I first converted the images to grayscale and then normalized them. Here are a few examples of both:

![alt text][im04]
![alt text][im05]

I implemented the LeNet model architecture and added a few more layers. My final architecture consisted of the final layers:

| Layer         		|     Description	        						| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   						| 
| Convolution	    	| 5x5 filter with 1x1 stride, valid padding, outputs 28x28x16	|
| RELU			|								|
|Max Pooling 		| 2x2 ksize with 2x2 stride, valid padding, outputs 14x14x16	|
| Convolution	    	| 5x5 filter with 1x1 stride, valid padding, outputs 14x14x64	|
| RELU			|								|
|Max Pooling 		| 2x2 ksize with 2x2 stride, valid padding, outputs 7x7x64	|
| Convolution	    	| 5x5 filter with 1x1 stride, valid padding, outputs 7x7x412	|
| RELU			|								|
| Fully connected 	| input – 412, output – 122 					| 
| RELU			|								|
| Dropout		| keep_prob=0.7						|
| Fully connected 	| input – 122, output – 84 					| 
| RELU			|								|
| Dropout		| keep_prob=0.7						|
| Fully connected 	| input – 84, output – 43 					| 

To train the model, I used an the following hyperparameters:
* Epoch = 30
*Batch size = 128
*Learning rate = 0.001
*Optimizer = Adam
*Loss function = Cross entropy
*Mu - 0
*Sigma - 0.1
*Dropout keep probability - 0.5

My final model results were:
* training set accuracy of 99.8%.
* validation set accuracy of 95.2%.
* test set accuracy of 94.4%.

### Testing the Model on New Images

Here are six German traffic signs that I found on the web:

![alt text][im06] ![alt text][im07] ![alt text][im08] 
![alt text][im09] ![alt text][im10] ![alt text][im11]

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 30 km/h      		| 30 km/h      									| 
| Bumpy Road     			| Bumpy Road  										|
| Ahead Only					| Ahead Only											|
| No vehicles	      		| Keep right					 				|
| Go straight or left			| Go straight or left							|
| General Caution		|General Caution					|


The model was able to correctly guess 5 of the 6 traffic signs, which gives an accuracy of 83%. This compares favourably to the accuracy on the test set of 95%

The top 5 softmax probabilities for each image are as follow:
TopKV2(values=array([[  1.00000000e+00,   0.00000000e+00,   0.00000000e+00,
          0.00000000e+00,   0.00000000e+00],
       [  1.00000000e+00,   0.00000000e+00,   0.00000000e+00,
          0.00000000e+00,   0.00000000e+00],
       [  1.00000000e+00,   0.00000000e+00,   0.00000000e+00,
          0.00000000e+00,   0.00000000e+00],
       [  9.99997854e-01,   2.13559588e-06,   2.41217553e-30,
          0.00000000e+00,   0.00000000e+00],
       [  1.00000000e+00,   0.00000000e+00,   0.00000000e+00,
          0.00000000e+00,   0.00000000e+00],
       [  1.00000000e+00,   0.00000000e+00,   0.00000000e+00,
          0.00000000e+00,   0.00000000e+00]], dtype=float32), indices=array([[ 1,  0,  2,  3,  4],
       [22,  0,  1,  2,  3],
       [35,  0,  1,  2,  3],
       [38, 15,  2,  0,  1],
       [37,  0,  1,  2,  3],
       [18,  0,  1,  2,  3]], dtype=int32))

The model predicted every image correctly with 100% probability but the 4th, for which it predicted keep right as the most probable output and then the correct one (No vehicles) as the second most probable. 
