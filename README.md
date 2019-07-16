 
# lsdy

### Installation

```sh
$ virtualenv lsdy
$ cd lsdy
$ source bin/active
$ git clone https://github.com/LSDkk5/lsdy
$ pip install -r src/REQUIREMENTS.txt
$ mkdir weights
$ curl https://pjreddie.com/media/files/yolov3.weights  --output 416.weights
$ mv 416.weights weights/416.weights

```
#### data/input - Put here images or videos


### Start detecting from image
```sh
$ python lsdy.py --image
```

### Start detecting from video
```sh
$ python lsdy.py --image
```