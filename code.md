## This is the source code of this project

The dataset used for the training is from bircatmcri, link: https://github.com/bircatmcri/MCIndoor20000

I used lablelImg by tzutalin to box the dataset, link: https://github.com/tzutalin/labelImg

And I used darkflow framework for training the model, link: https://github.com/veshitala/darkflow

### Preparation

Clone the darkflow framework

```python
!git clone https://github.com/veshitala/darkflow.git
```


Import some libraries

```python
%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np

from darkflow.net.build import TFNet
import cv2
```

### Training

Configure the hyperparameter and the dataset and annotation folder.

```python
option = { "model": "./cfg/tiny-yolo-new.cfg", 
           "batch": 16,
           "epoch": 400,
           "load": -1,
           "lr": 1e-4,
           "summary": "./",
           "gpu": 1.0,
           "train": True,
           "annotation": "./Annotation/",
           "dataset": "./Image/"}
           
tfnet = TFNet(option)
```

Train the model.

```python
tfnet.train()
```

### Test the model

Test on an image

```python
original_img = cv2.imread('PATH_IMAGE_TEST')
original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
results = tfnet.return_predict(original_img)
```

Box the image

```python
def boxing(original_img , predictions):
    newImage = np.copy(original_img)

    for result in predictions:
        top_x = result['topleft']['x']
        top_y = result['topleft']['y']

        btm_x = result['bottomright']['x']
        btm_y = result['bottomright']['y']

        confidence = result['confidence']
        label = result['label'] + " " + str(round(confidence, 3))
        
        if confidence > 0.1:
            newImage = cv2.rectangle(newImage, (top_x, top_y), (btm_x, btm_y), (255,0,0), 3)
            newImage = cv2.putText(newImage, label, (top_x, top_y-5), cv2.FONT_HERSHEY_COMPLEX_SMALL , 0.8, (0, 230, 0), 1, cv2.LINE_AA)
        
    return newImage
    
fig, ax = plt.subplots(figsize=(10, 5))
ax.imshow(boxing(original_img, results))
```
Here is the result.

![Test on image](https://github.com/Prakhosha/Computer-Vision-Stairs-Detection/blob/master/Demo/demo_image.png)

Test on a video

```python
cap = cv2.VideoCapture('PATH_INPUT_VIDEO')
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)   
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT) 

fourcc = cv2.VideoWriter_fourcc(*'DIVX')
out = cv2.VideoWriter('PATH_OUTPUT_VIDEO',fourcc, 20.0, (int(width), int(height)))

while(True):
    ret, frame = cap.read()
    
    if ret == True:
        frame = np.asarray(frame)      
        results = tfnet2.return_predict(frame)
        
        new_frame = boxing(frame, results)

        out.write(new_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break
        
cap.release()
out.release()
cv2.destroyAllWindows()
```

Here is the result

[![Test on video](https://github.com/Prakhosha/Computer-Vision-Stairs-Detection/blob/master/Demo/thumbnsil.jpg)](https://youtu.be/zr5mx4c9Bj8)
