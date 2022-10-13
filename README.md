# Image-Captioning-via-YOLOv5-EncoderDecoderwithAttention

#### Archived : This code was just a fun project! Neither it was propely tuned nor it is properly maintained! No Updates/correction expected! Focusing on other areas, I am not a vision expert. This code runs properly for most of the people, please check if images are getting populated properly for you if there is an error!**


Use original Flickr8K dataset. - https://www.kaggle.com/datasets/adityajn105/flickr8k 

PUT 'Images' directory and 'captions.txt' in the same directory as in root of this repo.


Attempt for Image Captioning using combination of object detection via YOLOv5 and Encoder Decoder LSTM model on Flickr8K dataset.

1. Run to make object crops via YOLOv5
```
python detect_object.py
```
2. Run to train - This just takes the Resnet embeddngs of object cropped images detected not any kind of text from YOLO labeller.
```
python train.py True
```
3. To evaluate on validation data
```
python train.py False
```
4. Sample predictions - 

![2.jpg](https://github.com/akjayant/Image-Captioning-via-YOLOv5-EncoderDecoderwithAttention/blob/main/test_images/2.jpg)

```
references -  [This is a black dog splashing in the water, A black lab with tags frolicks in the water ,A black dog running in the surf,The black dog runs through the water]

prediction- [['<SOS>'], ['a'], ['black'], ['dog'], ['is'], ['a'], ['a'], ['water'], ['.'], ['<EOS>']]
```

![1.jpg](https://github.com/akjayant/Image-Captioning-via-YOLOv5-EncoderDecoderwithAttention/blob/main/test_images/1.jpg)
```
references -  [A black dog and a spotted dog are fighting, A black dog and a tri-colored dog playing with each other on the road,
A black dog and a white dog with brown spots are staring at each other in the street,Two dogs of different breeds looking at each other on the road]

prediction- [['<SOS>'], ['a'], ['black'], ['and'], ['white'], ['dog'], ['is'], ['running'], ['through'], ['a'], ['.'], ['<EOS>']]
```
 5. Mean BLEU-4 score on validation data is quite low. Suggested Improvements : Use Adam and shuffling of data. Maybe minibatching. Increasing number of datapoints combining other datasets since 8k is quite low smaple size (No of parameters >> no of datapoints, not ideal for neural nets).


## Citation (Flickr8K Dataset)

Hodosh, Micah, Peter Young, and Julia Hockenmaier. "Framing image description as a ranking task: Data, models and evaluation metrics." Journal of Artificial Intelligence Research 47 (2013): 853-899.


