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
#import sklearn.manifold.LocallyLinearEmbedding as LLE


BUFFER_SIZE = 10000
BATCH_SIZE = 64
VOCAB_SIZE=1000
SEED = 100100

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
def load_main_model(model_path):
    VOCAB_SIZE=1000
    encoder = tf.keras.layers.experimental.preprocessing.TextVectorization(
        max_tokens=VOCAB_SIZE)
    dataset = tfds.load('imdb_reviews', as_supervised=True)
    train_dataset = dataset["train"]
    train_dataset = train_dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE).prefetch(tf.data.experimental.AUTOTUNE)
    encoder.adapt(train_dataset.map(lambda text, label: text))

    model = tf.keras.Sequential([
        encoder,
        tf.keras.layers.Embedding(
            input_dim=len(encoder.get_vocabulary())+2,
            output_dim=64,
            # Use masking to handle the variable sequence lengths
            mask_zero=True),
        tf.keras.layers.LSTM(64),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1)
    ])
    model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                  optimizer=tf.keras.optimizers.Adam(1e-4),
                  metrics=['accuracy'])
    model.load_weights(model_path)
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
    for k in range(0,len(tokens)):
        probs.append(model.predict(np.array([" ".join(tokens[:k+1])]))[0][0])
    pred = "POSITIVE" if probs[-1] >= 0 else "NEGATIVE"
    probs = np.diff(probs)
    colored_texts = [make_colored_text(token, p, probs)
                     for token, p in zip(tokens, probs)]
    return " ".join(colored_texts), pred


main_df = load_data("combined_sentiment_labelled.tsv")
# embeddings
embedding = load_pickled("train_embedding.pkl")
#tsned_space_raw_emb = load_pickled("tsne_space_raw_embedding.pkl")
#tsned_space_intermediate_emb = load_pickled("tsne_space_interm_embedding.pkl")
#tsned_space_proc_emb = load_pickled("tsne_space_proc_embedding.pkl")

# t-SNEs
#tsned_space_raw = load_pickled("tsne_space_raw.pkl")
#tsned_space_intermediate = load_pickled("tsne_space_interm.pkl")
#tsned_space_proc = load_pickled("tsne_space_proc.pkl")

np.random.seed(SEED)

st.write(main_df.head())

main_model = load_main_model("first_model/")

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

text = st.text_input("Type your review!")
if text != "":
    color_text, label = color_text(text, main_model)

    st.markdown(color_text, unsafe_allow_html=True)

    # add user's
    sentences = np.append(main_df["text"].values, text)
    probs = main_model.predict(sentences).reshape(-1).round(2)
    labels = ['Positive' if x else 'Negative'
              for x in (probs.reshape(-1) > 0)]
    labels[-1] = "User"
    emb = sen2vec_model.predict([text])[0]
    embedding = np.append(embedding, [emb], axis=0)

    tsne = TSNE(random_state=SEED, n_jobs=-1)
    tsned_space_raw_emb = tsne.fit_transform(sen2vec([[x] for x in sentences]).numpy().mean(axis=1))
    tsned_space_proc_emb = tsne.fit_transform(sen2vec_model.predict(sentences))
    tsned_space_intermediate_emb = tsne.fit_transform(embedding)

    tsne_plot_data = pd.DataFrame({
        'x_raw': tsned_space_raw_emb[:,0],
        'y_raw': tsned_space_raw_emb[:,1],
        'x_interm': tsned_space_intermediate_emb[:,0],
        'y_interm': tsned_space_intermediate_emb[:,1],
        'x_proc': tsned_space_proc_emb[:,0],
        'y_proc': tsned_space_proc_emb[:,1],
        'sentence': sentences,
        'prob': probs.astype(str),
        'pred': labels})

    selector_embs = alt.selection_interval(empty='all', encodings=['x', 'y'])
    row1_1, row1_2, row1_3 = st.beta_columns((1, 1, 1))
    with row1_1:
        words_tsned = alt.Chart(tsne_plot_data).mark_circle(size=200).encode(
            x = 'x_raw',
            y = 'y_raw',
            tooltip =[alt.Tooltip('sentence'), alt.Tooltip('prob')],
            color = alt.Color('pred', scale=alt.Scale(domain=['Negative', 'Positive', 'User'],
                                                      range=['red', 'green', 'blue'])),
            #opacity=alt.condition(selector_embs, 'opacity', alt.value(0.05), legend=None)
            opacity=alt.Opacity('prob', legend=None)
        ).properties(
            title='Raw sentences'
        ).add_selection(
            selector_embs
        )
        st.altair_chart(words_tsned)

    with row1_2:
        interm_tsned = alt.Chart(tsne_plot_data).mark_circle(size=200).encode(
            x = 'x_interm',
            y = 'y_interm',
            tooltip =[alt.Tooltip('sentence'), alt.Tooltip('prob')],
            color = alt.Color('pred', scale=alt.Scale(domain=['Negative', 'Positive', 'User'],
                                                      range=['red', 'green', 'blue'])),
            #opacity=alt.condition(selector_embs, 'opacity', alt.value(0.05), legend=None)
            opacity=alt.Opacity('prob', legend=None)
        ).properties(
            title='Intermediate state sentences'
        ).add_selection(
            selector_embs
        )
        st.altair_chart(interm_tsned)

    with row1_3:
        sentences_tsned = alt.Chart(tsne_plot_data).mark_circle(size=200).encode(
            x = 'x_proc',
            y = 'y_proc',
            tooltip =[alt.Tooltip('sentence'), alt.Tooltip('prob')],
            color = alt.Color('pred', scale=alt.Scale(domain=['Negative', 'Positive', 'User'],
                                                      range=['red', 'green', 'blue'])),
            #opacity=alt.condition(selector_embs, 'opacity', alt.value(0.05), legend=None)
            opacity=alt.Opacity('prob', legend=None)
        ).properties(
            title='Processed sentences'
        ).add_selection(
            selector_embs
        )
        st.altair_chart(sentences_tsned)


    distances = [cosine(emb, other) for other in embedding[:-1, :]]
    main_df["probs"] = probs[:-1] # note +1 user's label
    main_df["distance"] = distances
    sorted_ixs = np.argsort(distances)

    st.write("These are the probabilities assigned for your neighboring reviews",
             main_df.iloc[sorted_ixs, :])
    #st.write("These are the top 5 reviews that are the closest to yours: ",
    #         main_df.iloc[sorted_ixs, :].head(5))

    #st.write("These are the top 5 reviews that are the farthest to yours: ",
    #         main_df.iloc[sorted_ixs[::-1], :].head(5))
