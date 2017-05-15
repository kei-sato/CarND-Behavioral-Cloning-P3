**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[center-driving]: ./images/center-driving.gif "center-driving"
[curve]: ./images/curve.gif "curve"
[recovery]: ./images/recovery.gif "recovery"
[steerings]: ./images/steerings.png "steerings"
[center]: ./images/center.jpg "center"
[left]: ./images/left.jpg "left"
[right]: ./images/right.jpg "right"
[image1]: ./images/image1.jpg "image1"
[image1-flip]: ./images/image1-flip.jpg "image1-flip"

---

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup.md summarizing the results

Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 

```sh
python drive.py model.h5
```

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

My model is based on the [NVIDIA's model](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/) which contains several covolution layers followed by some fully connected layers.

I added a cropping layer (model.py line 39), a normalization layer (model.py line 40), and a dropout layer (model.py line 47) to this model.

The model was trained and validated on different data sets to ensure that the model was not overfitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

The model used an adam optimizer, so the learning rate was not tuned manually.

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road, and the driving smoothly around curves.

For details about how I created the training data, see the next section. 

#### 1. Solution Design Approach

My first step was using lenet model (model.py lines 22-34) with a training set which includes only one lap of driving. It failed at the middle of the first curve. So, I recorded another 2 attempts of the driving at the curve. But it did not work well. 

My next step was to use an another model, that was [NVIDIA's model](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/) introduced in the lesson of this project. It worked better, but it failed at the curve after the bridge. So I recorded another 2 attempts of the driving at that curve as well. Then it worked.

To combat the overfitting, I inserted a dropout layer to the model. Then I found the loss would decrease after 3 epochs, so I increased epochs from 3 to 5.

Finally, the vehicle is able to drive autonomously around the track without leaving the road in the simulator.

#### 2. Final Model Architecture

My final model (model.py lines 37-52) consisted of the following layers:

| Layer							| Description														| 
|:-----------------------------:|:-----------------------------------------------------------------:| 
| 1. Cropping2D					| removing 50 pixels from top, 20 pixels from bottom				| 
| 2. Lambda						| normalizing the values into the range [-0.5, 0.5]					|
| 3. Convolution2D				| 5x5 kernel, 2x2 stride, 24 filters 								|
| 4. Convolution2D				| 5x5 kernel, 2x2 stride, 36 filters								|
| 5. Convolution2D				| 5x5 kernel, 2x2 stride, 48 filters								|
| 6. Convolution2D				| 3x3 kernel, 1x1 stride, 64 filters								|
| 7. Convolution2D				| 3x3 kernel, 1x1 stride, 64 filters								|
| 9. Flatten					| 																	|
| 10. Dropout					| ratio 0.5															|
| 11. Dense						| outputs 100														|
| 12. Dense						| outputs 50														|
| 13. Dense						| outputs 10														|
| 14. Dense						| outputs 1															|


#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded one lap on track one using center lane driving. Here is an example image of center lane driving:

![alt text][center-driving]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn how to recover from the side of the road. This image shows what a recovery looks like:

![alt text][recovery]

Then I added some images forcusing on driving smoothly around some curves to avoid pop over the road while driving around curves.

![alt text][curve]

The number of data points here is 2,349. And here is the frequency of steering angles:

![alt text][steerings]

I also used images of the side cameras with a little correction of the steering angle. Here are the images taken by 3 cameras:

![alt text][center]
![alt text][left]
![alt text][right]

And to augment the data sat, I also flipped images and angles thinking that this would help the model to learn how to turn left/right even if the image at the moment was to turn opposite direction. It means the model could learn twice with one image. For example, here is an image that has then been flipped:

![alt text][image1]
![alt text][image1-flip]

The final number of data points is augmented up to 14,094 which is 6 times as large as it was.

I finally randomly shuffled the data set and put 20% of the data into a validation set.
