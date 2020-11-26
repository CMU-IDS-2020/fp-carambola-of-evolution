import pickle
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
        return pickle.load(r_obj)

@st.cache(allow_output_mutation=True)
def load_embedding(file_name):
    return np.load(file_name, allow_pickle=True)

@st.cache(allow_output_mutation=True)
def load_main_model():
    dataset = tfds.load('imdb_reviews', as_supervised=True)
    train_dataset = dataset["train"]
    train_dataset = train_dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE).prefetch(tf.data.experimental.AUTOTUNE)

    VOCAB_SIZE=5000
    encoder = tf.keras.layers.experimental.preprocessing.TextVectorization(
            max_tokens=VOCAB_SIZE)
    encoder.adapt(train_dataset.map(lambda text, label: text))

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
            return f"<span style='color:grey'>{text}</span>"
        elif p < 0:
            return f"<span style='color:red'>{text}</span>"
        else:
            return f"<span style='color:green'>{text}</span>"

    tokens = text.split(" ")
    probs = [0]
    for k in range(0, len(tokens)):
        probs.append(model.predict(np.array([" ".join(tokens[:k+1])]))[0][0])
    fin_prob = probs[-1]
    probs = np.diff(probs)
    colored_texts = [make_colored_text(token, p, probs)
                     for token, p in zip(tokens, probs)]
    return " ".join(colored_texts), fin_prob


st.title("Explaining RNNs")

main_df = load_data("combined_sentiment_labelled.tsv")

embedding = load_pickled("train_embedding.pkl")

np.random.seed(SEED)

main_model = load_main_model()

colored = load_pickled('colored.txt')

def sen2vec(x):
    return main_model.get_layer(name='embedding')(main_model.get_layer(name="text_vectorization")(x))

sen2vec_model = tf.keras.Sequential([
        main_model.get_layer(name="text_vectorization"),
        main_model.get_layer(name='embedding'),
        main_model.get_layer(name='lstm'),
        main_model.get_layer(name='dense')
    ])

sen2vec_model_interm = tf.keras.Sequential([
    main_model.get_layer(name="text_vectorization"),
    main_model.get_layer(name='embedding'),
    main_model.get_layer(name='lstm')
])


n_neighbor = st.slider(
    "Choose the number of neighboring reviews to find",
    min_value=50, max_value=len(main_df), value=50, step=50
    )

ixs = get_ixs(len(main_df), n_neighbor)
main_df = main_df.iloc[ixs, :]
embedding = embedding[ixs, :]



st.markdown('## Inference:')

def sample_inference():
    idx = np.random.randint(0, len(main_df), size=5)
    for i in idx:
        st.markdown(colored[i], unsafe_allow_html=True)

if st.button('Sample another reviews'):
    sample_inference()
else:
    sample_inference()





st.markdown('## Training:')

if st.button('Sample random review'):
    review = main_df.iloc[np.random.randint(0, len(main_df))].text
    text = st.text_input("Type your review!", review)
else:
    text = st.text_input("Type your review!")

if text != "":

    sentences = np.append(main_df["text"].values, text)

    for i in range(6):
        main_model.load_weights(f'./training/cp-000{i}.ckpt')
        pred = color_text(text, model=main_model)
        st.markdown(f"Epoch {i}" + " | " +
                    ("NEG" if pred[1] < 0 else "POS") + " | " +
                    str(probability(pred[1])) + " | " +
                    pred[0],
                    unsafe_allow_html=True)

    for i in range(6):
        main_model.load_weights(f'./training/cp-000{i}.ckpt')
        probs = main_model.predict(sentences).reshape(-1).round(2)
        labels = ['Positive' if x else 'Negative'
                  for x in (probs.reshape(-1) > 0)]
        labels[-1] = "User"
        emb = sen2vec_model.predict([text])[0]
        embedding_ep = np.append(embedding, [emb], axis=0)

        tsne = TSNE(random_state=SEED, n_jobs=-1)
        tsned_space_raw_emb = tsne.fit_transform(sen2vec([[x] for x in sentences]).numpy().mean(axis=1))
        tsned_space_proc_emb = tsne.fit_transform(sen2vec_model.predict(sentences))
        tsned_space_intermediate_emb = tsne.fit_transform(embedding_ep)

        tsne_plot_data = pd.DataFrame({
            'x_raw': tsned_space_raw_emb[:,0],
            'y_raw': tsned_space_raw_emb[:,1],
            'x_interm': tsned_space_intermediate_emb[:,0],
            'y_interm': tsned_space_intermediate_emb[:,1],
            'x_proc': tsned_space_proc_emb[:,0],
            'y_proc': tsned_space_proc_emb[:,1],
            'sentence': sentences,
            'opacity': np.abs(probs),
            'prob': probability(probs).astype(str),
            'pred': labels})

        selector_embs = alt.selection_interval(empty='all', encodings=['x', 'y'])

        words_tsned = alt.Chart(tsne_plot_data).mark_circle(size=200).encode(
            x = 'x_raw',
            y = 'y_raw',
            tooltip =[alt.Tooltip('sentence'), alt.Tooltip('prob')],
            color = alt.Color('pred', scale=alt.Scale(domain=['Negative', 'Positive', 'User'],
                                                      range=['red', 'green', 'blue'])),
            opacity=alt.condition(selector_embs, 'opacity', alt.value(0.05), legend=None)
        ).properties(
            title='Raw sentences',
            height=HEIGHT,
            width=WIDTH
        ).add_selection(
            selector_embs
        )

        interm_tsned = alt.Chart(tsne_plot_data).mark_circle(size=200).encode(
            x = 'x_interm',
            y = 'y_interm',
            tooltip =[alt.Tooltip('sentence'), alt.Tooltip('prob')],
            color = alt.Color('pred', scale=alt.Scale(domain=['Negative', 'Positive', 'User'],
                                                      range=['red', 'green', 'blue'])),
            opacity=alt.condition(selector_embs, 'opacity', alt.value(0.05), legend=None)
        ).properties(
            title='Intermediate state sentences',
            height=HEIGHT,
            width=WIDTH
        ).add_selection(
            selector_embs
        )

        sentences_tsned = alt.Chart(tsne_plot_data).mark_circle(size=200).encode(
            x = 'x_proc',
            y = 'y_proc',
            tooltip =[alt.Tooltip('sentence'), alt.Tooltip('prob')],
            color = alt.Color('pred', scale=alt.Scale(domain=['Negative', 'Positive', 'User'],
                                                      range=['red', 'green', 'blue'])),
            opacity=alt.condition(selector_embs, 'opacity', alt.value(0.05), legend=None)
        ).properties(
            title='Processed sentences',
            height=HEIGHT,
            width=WIDTH
        ).add_selection(
            selector_embs
        )

        st.altair_chart(words_tsned | interm_tsned | sentences_tsned)

#    distances = [cosine(emb, other) for other in embedding[:-1, :]]
#    main_df["probs"] = probs[:-1] # note +1 user's label
#    main_df["distance"] = distances
#    sorted_ixs = np.argsort(distances)[:5]

#    st.write("These are the probabilities assigned for your neighboring reviews:")
#    for _, row in main_df.iloc[sorted_ixs, :].iterrows():
#        st.write("* " + row.text)