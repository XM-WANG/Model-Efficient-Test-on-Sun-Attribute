# Model Efficient Test on Sun Attribute Dataset
Multi-label Classification for Sun Attribute Dataset

Introduction:
-------
We test **Res-Net**, **VGG-Net** and **Dense-Net** on Sun Attribute Dataset. We try different hyper parameters,like *learning rate*, *depth* of model, *batch-size*, *epoch* etc, on models respectively and find the best parameters group. The results and conclusion is shown in `report.pdf`.


Quick Start:
-------
1. Download the dataset from https://cs.brown.edu/~gen/Attributes/SUNAttributeDB_Images.tar.gz
2. Make a dir named "images" in "data/" and put the images in it
3. Run train.py to train a model (default is vgg16-based model)
4. Run test.py to test your trained model and get recall and precision

Code Structure:
-------
**TODO**: Many files should be merged( eg. `trai11.py` and `train2.py`). These files are chaos because of the pressing deadline.

* `train.py`: a script for training
* `test.py`: a script for testing
* `classifier/dataset`: parse the SUN Attribute Database (In most cases you do NOT need to make changes to it)
* `classifier/models`: different base models, including reset and vgg
* `classifier/utils`: some basic codes, including metrics, data transformer, data pre-processer and so on (In most cases you do NOT need to make changes to it)
* `classifier/trainer.py`: a trainer for training (In most cases you do NOT need to make changes to it)
* `calssidier/evaluator.py`: an evaluator for evaluation and testing (In most cases you do NOT need to make changes to it)
