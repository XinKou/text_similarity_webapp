# Text Similarity Web App

It is a flask web App that interfaces with a trained neural netowork to predict the similarity between two short texts. This web app provides two text boxes for users to enter the text they wish to compare and returns with an answer such as *very similar*,  *similar* or *not similar*.  

To predict text similarity, an **LSTM** network with **word2vec** and **GloVe** as the word embeddings is trained on more than 400K [Quora Question Pairs][]. This model was trained on AWS EC2 using GPU for performance. Due to the long traning time of the model (especially for CPU), I include an already trained LSTM model. 
[quora question pairs]: https://www.kaggle.com/c/quora-question-pairs/data
## Setup Environment on Local Machine

* Verify you have anaconda or miniconda installed by typing `conda` in your terminal window.




### Install Packages (Flask and other dependencies) - Steps not included



## Run the App
python run.py


### Test App

1. Open Browser:  [http://localhost:5000](http://localhost:5000)
2. Type two short texts and click `sumbit`

## Credits:

* Web app template from https://github.com/sampathweb/cc-iris-flask-app and also inspired by the amazon review sentiment analysis web app from Rui Wang and Xiaojuan Tian: https://github.com/wangruinju/Amazon_Review_Sentiment_Analysis.
* LSTM model inspired by the [kaggle kernel][].

[kaggle kernel]: https://www.kaggle.com/lystdo/lstm-with-word2vec-embeddings


