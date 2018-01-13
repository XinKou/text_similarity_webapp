import logging
import json

from flask import render_template
from flask_wtf import Form
from wtforms import fields
from wtforms.validators import Required
from wtforms.widgets import TextArea

from . import app

from sklearn.externals import joblib
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
import re
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from string import punctuation



logger = logging.getLogger('app')

class PredictForm(Form):
    """Fields for Predict"""
    # sepal_length = fields.DecimalField('Sepal Length:', places=2, validators=[Required()])
    # sepal_width = fields.DecimalField('Sepal Width:', places=2, validators=[Required()])
    # petal_length = fields.DecimalField('Petal Length:', places=2, validators=[Required()])
    # petal_width = fields.DecimalField('Petal Width:', places=2, validators=[Required()])
    text_1 = fields.TextAreaField('Text 1:', validators=[Required()])
    text_2 = fields.TextAreaField('Text 2:', validators=[Required()])

    submit = fields.SubmitField('Submit')


def text_to_wordlist(text, remove_stopwords=False, stem_words=False):
    # Clean the text, with the option to remove stopwords and to stem words.
    
    # Convert words to lower case and split them
    text = text.lower().split()

    # Optionally, remove stop words
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        text = [w for w in text if not w in stops]
    
    text = " ".join(text)

    # Clean the text
    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "cannot ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r",", " ", text)
    text = re.sub(r"\.", " ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\/", " ", text)
    text = re.sub(r"\^", " ^ ", text)
    text = re.sub(r"\+", " + ", text)
    text = re.sub(r"\-", " - ", text)
    text = re.sub(r"\=", " = ", text)
    text = re.sub(r"'", " ", text)
    text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
    text = re.sub(r":", " : ", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r" b g ", " bg ", text)
    text = re.sub(r" u s ", " american ", text)
    text = re.sub(r"\0s", "0", text)
    text = re.sub(r" 9 11 ", "911", text)
    text = re.sub(r"e - mail", "email", text)
    text = re.sub(r"j k", "jk", text)
    text = re.sub(r"\s{2,}", " ", text)
    
    # Optionally, shorten words to their stems
    if stem_words:
        text = text.split()
        stemmer = SnowballStemmer('english')
        stemmed_words = [stemmer.stem(word) for word in text]
        text = " ".join(stemmed_words)
    
    # Return a list of words
    return(text)

tokenizer = joblib.load('models/tokenizer.pkl')
model = load_model('models/lstm.h5')
model._make_predict_function()

def make_prediction(s1,s2):
    #print(s1,s2)
    s1 = text_to_wordlist(s1)
    s2 = text_to_wordlist(s2)
    s1 = tokenizer.texts_to_sequences([s1])
    s2 = tokenizer.texts_to_sequences([s2])
    s1 = pad_sequences(s1, maxlen=30)
    s2 = pad_sequences(s2, maxlen=30)

    return (model.predict([s1, s2])+model.predict([s2, s1]))/2

@app.route('/', methods=('GET', 'POST'))
def index():
    """Index page"""
    form = PredictForm()
    pred= None

    if form.validate_on_submit():
        # store the submitted values
        submitted_data = form.data

        # Retrieve values from form
        text_1 = submitted_data['text_1']
        text_2 = submitted_data['text_2']
        

    
        print(make_prediction(text_1, text_2))
        # set 0.2 as the threshold being similar
        if make_prediction(text_1, text_2) >= 0.4:
            pred = 'very similar'
        elif make_prediction(text_1, text_2) >= 0.2:
            pred = 'similar'
        else:
            pred = 'not similar'


    return render_template('index.html',
        form=form,
        prediction=pred)
