# Image-Captioning-via-YOLOv5-EncoderDecoderwithAttention
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
4 . Does okayish. Mean BLEU-4 score on validation data = 0.42  (0r 42 on a scale of 100)
5. Sample predictions - 

![2.jpg](https://github.com/akjayant/Image-Captioning-via-YOLOv5-EncoderDecoderwithAttention/blob/main/test_images/2.jpg)

```
correct -  [This is a black dog splashing in the water, A black lab with tags frolicks in the water ,A black dog running in the surf,The black dog runs through the water]

prediction- [['<SOS>'], ['a'], ['black'], ['dog'], ['is'], ['a'], ['a'], ['water'], ['.'], ['<EOS>']]
```

![1.jpg](https://github.com/akjayant/Image-Captioning-via-YOLOv5-EncoderDecoderwithAttention/blob/main/test_images/1.jpg)
```
correct -  A black dog and a spotted dog are fighting

prediction- [['<SOS>'], ['a'], ['black'], ['and'], ['white'], ['dog'], ['is'], ['running'], ['through'], ['a'], ['.'], ['<EOS>']]
```


