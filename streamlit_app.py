import re
#import time
import nltk
import pickle
import string
import numpy as np
import pandas as pd
import altair as alt
import streamlit as st
import tensorflow as tf
from sklearn.manifold import TSNE
from tensorflow.keras import layers
import tensorflow_datasets as tfds
from scipy.spatial.distance import cosine

HEIGHT = 200
WIDTH = 200

BUFFER_SIZE = 10000
BATCH_SIZE = 64
SEED = 100100


# for processing text
try:
    nltk.data.find("stopwords")
except:
    nltk.download("stopwords", quiet = True)
try:
    nltk.data.find("wordnet")
except:
    nltk.download("wordnet", quiet = True)
try:
    nltk.data.find("punkt")
except:
    nltk.download("punkt", quiet = True)
try:
    nltk.data.find('averaged_perceptron_tagger')
except:
    nltk.download('averaged_perceptron_tagger', quiet = True)

lemmatizer = nltk.stem.wordnet.WordNetLemmatizer()
english_stopwords = set(nltk.corpus.stopwords.words('english'))

@st.cache(allow_output_mutation=True)
def process_text(text):
    def get_pos(tag):
        if tag.startswith("J"):
            return "a"
        elif tag.startswith("V"):
            return "v"
        elif tag.startswith("R"):
            return "r"
        else:
            return "n"

    text = text.replace("<br />", "")
    text = text.replace("\'", "'")

    text = re.sub(r"'s", "", text.lower())
    text = re.sub(r"([a-z0-9]+)'([^s])", r"\1\2", text)
    text = re.sub(rf"[^{string.ascii_letters}0-9]", " ", text)


    tokenized = []
    for token in nltk.word_tokenize(text):
        token, tag = nltk.pos_tag([token])[0]
        t = lemmatizer.lemmatize(token, pos=get_pos(tag))
        if t not in english_stopwords and len(t) > 1:
            tokenized.append(t)
    return " ".join(tokenized)


def predict(model, sentences):
    return model.predict(np.array([process_text(s) for s in sentences]))


def probability(x):
    return np.round(np.abs(2 * (1 / (1 + np.exp(-x)) - 0.5)), 2)

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    GREEN = '\033[92m'
    WARNING = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'



@st.cache(allow_output_mutation=True)
def load_data(file_name):
    data = pd.read_csv(file_name, index_col=None, sep="\t")
    return data

@st.cache(allow_output_mutation=True)
def load_pickled(file_name):
    with open(file_name, "rb") as r_obj:
        return pickle.load(r_obj, encoding="utf-8")

@st.cache(allow_output_mutation=True)
def load_embedding(file_name):
    return np.load(file_name, allow_pickle=True)

#@st.cache(allow_output_mutation=True)
def load_main_model():
    #dataset = tfds.load('imdb_reviews', as_supervised=True)
    X_train = load_pickled("v.pkl")

    VOCAB_SIZE=10000
    encoder = tf.keras.layers.experimental.preprocessing.TextVectorization(
            max_tokens=VOCAB_SIZE)
    encoder.adapt(X_train)

    model = tf.keras.Sequential([
        encoder,
        tf.keras.layers.Embedding(
            input_dim=len(encoder.get_vocabulary()) + 2,
            output_dim=64,
            # Use masking to handle the variable sequence lengths
            mask_zero=True),
        tf.keras.layers.LSTM(64),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(1)
    ])

    model.compile()

    return model

@st.cache(allow_output_mutation=True)
def get_ixs(n_train, n_neighbor):
    return np.random.choice(range(n_train), n_neighbor, replace=False)

def color_text(text, model):
    def make_colored_text(text, p, probs):
        if abs(p / max(np.abs(probs))) < 0.1:
            return f"<span style='color:grey; opacity:0.3'>{text}</span>"
        elif p < 0:
            return f"<span style='color:red; opacity:{abs(p / max(np.abs(probs))) + 0.2}'>{text}</span>"
        else:
            return f"<span style='color:green; opacity:{abs(p / max(np.abs(probs))) + 0.2}'>{text}</span>"

    tokens = text.split()
    probs = [0]
    for k in range(0, len(tokens)):
        proc_text = process_text(" ".join(tokens[:k+1]))
        if proc_text == "":
            probs.append(probs[-1])
        else:
            probs.append(
                predict(model, np.array([proc_text]))[0][0]
            )
    fin_prob = probs[-1]
    probs = np.diff(probs)
    colored_texts = [make_colored_text(token, p, probs)
                     for token, p in zip(tokens, probs)]
    return " ".join(colored_texts), fin_prob

st.markdown("<h1 style='text-align: center;'>Explaining Recurrent Neural Networks</h1>", unsafe_allow_html=True)

st.markdown("<h3 style='text-align: right; color: gray;'>Made by Clay and Ihor</h3>", unsafe_allow_html=True)

st.write('In this interactive app you will be able to explore why RNN produce one or another output and what makes difference for the model in the input. ' 
         + 'The model that you will explore is a simple RNN that was built on IMDB reviews binary classification dataset (positive or negative review).')
st.write('The model consists from embedding layer, LSTM, single hidden dense layer with 64 neurons with ReLu activation and dense output layer with a single neuron.')

st.write('Proposed framework is illustrated on specific considerably simple model, however, it is general and can trivially be extended to larger and more complex models.')

main_df = load_data("combined_sentiment_labelled.tsv")

main_model = load_main_model()

colored = load_pickled('colored.txt')

def sen2vec(x):
    x = np.array([[process_text(xx[0])] for xx in x])
    return main_model.get_layer(index=1)(main_model.get_layer(index=0)(x))

sen2vec_model = tf.keras.Sequential([
        main_model.get_layer(index=0),
        main_model.get_layer(index=1),
        main_model.get_layer(index=2),
        main_model.get_layer(index=4)
    ])

sen2vec_model_interm = tf.keras.Sequential([
    main_model.get_layer(index=0),
    main_model.get_layer(index=1),
    main_model.get_layer(index=2)
])


# n_neighbor = st.slider(
#    "Choose the number of neighboring reviews to find",
#    min_value=50, max_value=len(main_df), value=50, step=50
#    )
n_neighbor = 500 #fix

#ixs = get_ixs(len(main_df), n_neighbor)
ixs = list(range(n_neighbor))
main_df = main_df.iloc[ixs, :]
#embedding = embedding[ixs, :]



st.markdown('## Inference:')

st.write('Firstly, let\'s try to get some insight how model works by observing which words contribute to decision whether to output positive or negative.')
st.write('Below you see five sampled reviews from the example dataset with prediction, confidence of the prediction and visualized word impacts. '
         + 'Color represents positive (green), negative (red) or neutral (grey) impact on models prediction, namely how models prediction and confidence changed after seeing that word. ' 
         + 'Opacity represents strength of impact - the higher the opacity, the more impact that word had!')
            
def sample_inference():
    idx = np.random.randint(0, len(colored), size=5)
    st.markdown(f'---------------------------')
    for i in idx:
        st.markdown(colored[i], unsafe_allow_html=True)
    st.markdown(f'---------------------------')

if st.button('Sample another reviews'):
    sample_inference()
else:
    sample_inference()





st.markdown('## Training:')

st.write('Now let\'s see how model arrived at such decision by visualizing its training process. '
         + 'In this part we will be working with a single sentence. Type your own or click a button to sample a random one!')
            
if st.button('Sample random review'):
    review = main_df.iloc[np.random.randint(0, len(main_df))].text
    text = st.text_input("Or type your review!", review)
else:
    text = st.text_input("Or type your review!", "This application is really cool and authors are great!")

if text != "":

    st.write('Firstly, we will provide same type of visualization for the review over several epochs. '
             + 'Observe the patterns in changes of the models confidence and how each word impacts the prediction.')
    
    sentences = np.append(main_df["text"].values, text)
    st.markdown(f'---------------------------')
    for i in range(0, 11, 2):
        main_model.load_weights(f"epoch{i}/")
        pred = color_text(text, model=main_model)
        st.markdown(f"Epoch {i}" + " | " +
                    ("NEG" if pred[1] < 0 else "POS") + " | " +
                    str(probability(pred[1])) + " | " +
                    pred[0],
                    unsafe_allow_html=True)
    st.markdown(f'---------------------------')
    
    st.write('Now let\'s visualize feature space and how it is transformed while being passed through models layers.')
    st.write('The leftmost plot is learned sentence embedding, '
             + 'the middle one is output of embeddings being passed through LSTM '
             + 'and the rightmost one is the output of LSTM output being passed through dense layer.')
    st.write('Charts are interactive in two ways - review text and probability will be shown on cursor hover and it\'s possible to select only subset of data by dragging a rectangle with mouse.')
    st.write('Note that originally all feature spaces are of high dimensionality and we approximate them for visualization with Isomap.')
    
    #for i in range(0, 11, 2):
    for i in [0, 4, 10]:
        st.markdown(f'#### Epoch {i}')
        isomap_raw = load_pickled(f'./embedding/isomap-raw-{i}.pkl')
        isomap_intermediate = load_pickled(f'./embedding/isomap-intermediate-{i}.pkl')
        isomap_proc = load_pickled(f'./embedding/isomap-proc-{i}.pkl')


        probs = predict(main_model, sentences).reshape(-1).round(2)
        labels = ['Positive' if x else 'Negative'
                  for x in (probs.reshape(-1) > 0)]
        labels[-1] = "User"

        raw_emb_text = isomap_raw.transform(
            sen2vec([[text]]).numpy().mean(axis=1)
            )
        isomap_raw_emb = np.append(isomap_raw.embedding_, raw_emb_text, axis=0)

        intermediate_emb_text = isomap_intermediate.transform(
            predict(sen2vec_model_interm, [text])
            )
        isomap_intermediate_emb = np.append(isomap_intermediate.embedding_,
                                          intermediate_emb_text,
                                          axis=0)

        proc_emb_text = isomap_proc.transform(predict(sen2vec_model, [text]))
        isomap_proc_emb = np.append(isomap_proc.embedding_, proc_emb_text, axis=0)
        plot_data = pd.DataFrame({
            'x_raw': isomap_raw_emb[:,0],
            'y_raw': isomap_raw_emb[:,1],
            'x_interm': isomap_intermediate_emb[:,0],
            'y_interm': isomap_intermediate_emb[:,1],
            'x_proc': isomap_proc_emb[:,0],
            'y_proc': isomap_proc_emb[:,1],
            'sentence': sentences,
            'opacity': np.abs(probs),
            'prob': probability(probs).astype(str),
            'pred': labels})

        selector_embs = alt.selection_interval(empty='all', encodings=['x', 'y'])

        words_tsned = alt.Chart(plot_data).mark_circle(size=200).encode(
            x = 'x_raw',
            y = 'y_raw',
            tooltip =[alt.Tooltip('sentence'), alt.Tooltip('prob')],
            color = alt.Color('pred', scale=alt.Scale(domain=['Negative', 'Positive', 'User'],
                                                      range=['red', 'green', 'blue']), legend=alt.Legend(symbolOpacity=1)),
            opacity=alt.condition(selector_embs, 'opacity', alt.value(0.05), legend=None)
        ).properties(
            title='Raw sentences',
            height=HEIGHT,
            width=WIDTH
        ).add_selection(
            selector_embs
        )

        interm_tsned = alt.Chart(plot_data).mark_circle(size=200).encode(
            x = 'x_interm',
            y = 'y_interm',
            tooltip =[alt.Tooltip('sentence'), alt.Tooltip('prob')],
            color = alt.Color('pred', scale=alt.Scale(domain=['Negative', 'Positive', 'User'],
                                                      range=['red', 'green', 'blue']), legend=alt.Legend(symbolOpacity=1)),
            opacity=alt.condition(selector_embs, 'opacity', alt.value(0.05), legend=None)
        ).properties(
            title='Intermediate state sentences',
            height=HEIGHT,
            width=WIDTH
        ).add_selection(
            selector_embs
        )

        sentences_tsned = alt.Chart(plot_data).mark_circle(size=200).encode(
            x = 'x_proc',
            y = 'y_proc',
            tooltip =[alt.Tooltip('sentence'), alt.Tooltip('prob')],
            color = alt.Color('pred', scale=alt.Scale(domain=['Negative', 'Positive', 'User'],
                                                      range=['red', 'green', 'blue']), legend=alt.Legend(symbolOpacity=1)),
            opacity=alt.condition(selector_embs, 'opacity', alt.value(0.05), legend=None)
        ).properties(
            title='Processed sentences',
            height=HEIGHT,
            width=WIDTH
        ).add_selection(
            selector_embs
        )

        st.altair_chart(words_tsned | interm_tsned | sentences_tsned)
