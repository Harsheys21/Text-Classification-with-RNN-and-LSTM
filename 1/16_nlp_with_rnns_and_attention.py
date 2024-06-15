#!/usr/bin/env python
# coding: utf-8

# **Chapter 16 – Natural Language Processing with RNNs and Attention**

# _This notebook contains all the sample code in chapter 16._

# <table align="left">
#   <td>
#     <a target="_blank" href="https://colab.research.google.com/github/jflanigan/handson-ml2/blob/master/16_nlp_with_rnns_and_attention.ipynb"><img src="https://www.tensorflow.org/images/colab_logo_32px.png" />Run in Google Colab</a>
#   </td>
# </table>

# # Setup

# First, let's import a few common modules, ensure MatplotLib plots figures inline and prepare a function to save the figures. We also check that Python 3.5 or later is installed (although Python 2.x may work, it is deprecated so we strongly recommend you use Python 3 instead), as well as Scikit-Learn ≥0.20 and TensorFlow ≥2.0.

# In[10]:


# Python ≥3.5 is required
import sys
assert sys.version_info >= (3, 5)

# Scikit-Learn ≥0.20 is required
import sklearn
assert sklearn.__version__ >= "0.20"

try:
    # %tensorflow_version only exists in Colab.
    get_ipython().run_line_magic('tensorflow_version', '2.x')
    get_ipython().system('pip install -q -U tensorflow-addons')
    IS_COLAB = True
except Exception:
    IS_COLAB = False

# TensorFlow ≥2.0 is required
import tensorflow as tf
from tensorflow import keras
assert tf.__version__ >= "2.0"

if not tf.test.is_gpu_available():
    print("No GPU was detected. LSTMs and CNNs can be very slow without a GPU.")
    if IS_COLAB:
        print("Go to Runtime > Change runtime and select a GPU hardware accelerator.")

# Common imports
import numpy as np
import os

# to make this notebook's output stable across runs
np.random.seed(42)
tf.random.set_seed(42)

# To plot pretty figures
#get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)

# Where to save the figures
PROJECT_ROOT_DIR = "."
CHAPTER_ID = "nlp"
IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID)
os.makedirs(IMAGES_PATH, exist_ok=True)

def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = os.path.join(IMAGES_PATH, fig_id + "." + fig_extension)
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)


# # Text Classification with RNNs

# In[11]:


tf.random.set_seed(42)


# You can load the IMDB dataset easily:

# In[12]:


(X_train, y_test), (X_valid, y_test) = keras.datasets.imdb.load_data()


# In[13]:


X_train[0][:10]


# In[14]:


word_index = keras.datasets.imdb.get_word_index()
id_to_word = {id_ + 3: word for word, id_ in word_index.items()}
for id_, token in enumerate(("<pad>", "<sos>", "<unk>")):
    id_to_word[id_] = token
" ".join([id_to_word[id_] for id_ in X_train[0][:10]])


# In[15]:


import tensorflow_datasets as tfds

datasets, info = tfds.load("imdb_reviews", as_supervised=True, with_info=True)


# In[16]:


datasets.keys()


# In[17]:


train_size = info.splits["train"].num_examples
test_size = info.splits["test"].num_examples


# In[18]:


train_size, test_size


# In[19]:


for X_batch, y_batch in datasets["train"].batch(2).take(1):
    for review, label in zip(X_batch.numpy(), y_batch.numpy()):
        print("Review:", review.decode("utf-8")[:200], "...")
        print("Label:", label, "= Positive" if label else "= Negative")
        print()


# In[20]:


def preprocess(X_batch, y_batch):
    X_batch = tf.strings.substr(X_batch, 0, 300)
    X_batch = tf.strings.regex_replace(X_batch, rb"<br\s*/?>", b" ")
    X_batch = tf.strings.regex_replace(X_batch, b"[^a-zA-Z']", b" ")
    X_batch = tf.strings.split(X_batch)
    return X_batch.to_tensor(default_value=b"<pad>"), y_batch


# In[21]:


preprocess(X_batch, y_batch)


# In[22]:


from collections import Counter

vocabulary = Counter()
for X_batch, y_batch in datasets["train"].batch(32).map(preprocess):
    for review in X_batch:
        vocabulary.update(list(review.numpy()))


# In[23]:


vocabulary.most_common()[:3]


# In[24]:


len(vocabulary)


# In[25]:


vocab_size = 10000
truncated_vocabulary = [
    word for word, count in vocabulary.most_common()[:vocab_size]]


# In[26]:


word_to_id = {word: index for index, word in enumerate(truncated_vocabulary)}
for word in b"This movie was faaaaaantastic".split():
    print(word_to_id.get(word) or vocab_size)


# In[27]:


words = tf.constant(truncated_vocabulary)
word_ids = tf.range(len(truncated_vocabulary), dtype=tf.int64)
vocab_init = tf.lookup.KeyValueTensorInitializer(words, word_ids)
num_oov_buckets = 1000
table = tf.lookup.StaticVocabularyTable(vocab_init, num_oov_buckets)


# In[28]:


table.lookup(tf.constant([b"This movie was faaaaaantastic".split()]))


# In[36]:


def encode_words(X_batch, y_batch):
    return table.lookup(X_batch), y_batch

train_set = datasets["train"].repeat().batch(32).map(preprocess)
test_set = datasets["test"].repeat().batch(32).map(preprocess)

train_set = train_set.map(encode_words).prefetch(1)
test_set = test_set.map(encode_words).prefetch(1)


# In[30]:


for X_batch, y_batch in train_set.take(1):
    print(X_batch)
    print(y_batch)


# In[38]:


embed_size = 16
hidden_dim = 64
dropout = 0.5
nonlinearity = 'tanh'
learning_rate = 1e-3

model = keras.models.Sequential([
    keras.layers.Embedding(vocab_size + num_oov_buckets, embed_size,
                           mask_zero=True, # not shown in the book
                           input_shape=[None]),
    keras.layers.SimpleRNN(hidden_dim, activation=nonlinearity, dropout=dropout),
    keras.layers.Dense(1, activation="sigmoid")
])
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
history = model.fit(train_set, steps_per_epoch=train_size // 32, epochs=20)


# In[37]:


# Train and test accuracy
print("Train set\n",model.evaluate(train_set, steps=train_size // 32, batch_size=32))

print("Test set\n",model.evaluate(test_set, steps=test_size // 32, batch_size=32))


# ## Text Classification with LSTMs

# In[39]:


model = keras.models.Sequential([
    keras.layers.Embedding(vocab_size + num_oov_buckets, embed_size,
                           mask_zero=True, # not shown in the book
                           input_shape=[None]),
    keras.layers.LSTM(hidden_dim, activation=nonlinearity, dropout=dropout),
    keras.layers.Dense(1, activation="sigmoid")
])
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
history = model.fit(train_set, steps_per_epoch=train_size // 32, epochs=20)


# In[40]:


# Train and test accuracy
print("Train set\n",model.evaluate(train_set, steps=train_size // 32, batch_size=32))

print("Test set\n",model.evaluate(test_set, steps=test_size // 32, batch_size=32))

