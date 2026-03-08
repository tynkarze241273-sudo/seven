# import json
# import streamlit as st
# import pandas as pd  # type: ignore
# from pycaret.clustering import load_model, predict_model  # type: ignore
# import plotly.express as px  # type: ignore

# TE LINIE MUSZĄ BYĆ PIERWSZE W CAŁYM PLIKU, PRZED WSZYSTKIM
import os
os.environ['PYCARET_NO_CDE'] = '1'
os.environ['SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL'] = 'True'

import warnings
warnings.filterwarnings("ignore")

# NA DRUGIM MIEJSCU IMPORTUJEMY NAJPIERW PYCARET, NIGDY NA ODWROT
from pycaret.clustering import ClusteringExperiment
from pycaret.utils import load_model, save_model

# DOPIERO NA SAMYM KOŃCU IMPORTUJEMY STREAMLIT
import streamlit as st
import pandas as pd

DATA = 'welcome_survey_simple_v2.csv'

MODEL_NAME = 'welcome_survey_clustering_pipeline_v2'

CLUSTER_NAMES_AND_DESCRIPTIONS = 'welcomesurvey_cluster_names_and_descriptions_v2.json'

@st.cache_resource
def get_cluster_names_and_descriptions():
    with open(CLUSTER_NAMES_AND_DESCRIPTIONS, "r", encoding='utf-8') as f:
        return json.loads(f.read())
    
@st.cache_resource
def get_model():
    return load_model(MODEL_NAME)


@st.cache_resource
def get_all_participants():
    model=get_model()
    all_df = pd.read_csv(DATA, sep=';')
    df_with_clusters = predict_model(model, data=all_df)

    return df_with_clusters

with st.sidebar:
    st.header("Powiedz nam coś o sobie")
    st.markdown("Pomożemy Ci znaleźć osoby, które mają podobne zainteresowania")
    age = st.selectbox("Wiek", ['<18', '25-34', '45-54', '35-44', '18-24', '>=65', '55-64', 'unknown'])
    edu_level = st.selectbox("Wykształcenie", ['Podstawowe', 'Średnie', 'Wyższe'])
    fav_animals = st.selectbox("Ulubione zwierzęta", ['Brak ulubionych', 'Psy', 'Koty', 'Inne', 'Koty i Psy'])
    fav_place = st.selectbox("Ulubione miejsce", ['Nad wodą', 'W lesie', 'W górach', 'Inne'])
    gender = st.radio("Płeć", ['Mężczyzna', 'Kobieta'])

    person_df = pd.DataFrame([
        {
            'age': age,
            'edu_level': edu_level,
            'fav_animals': fav_animals,
            'fav_place': fav_place,
            'gender': gender
        }
    ])
# st.write("Wybrane dane:")
# st.dataframe(person_df, hide_index=True)

# all_df = get_all_participants()
# st.write("Przykładowe osoby z bazy:")
# st.dataframe(all_df.sample(10), hide_index=True)

model = get_model() # wczytujemy model
all_df = get_all_participants() # wczytujemy uźytkowników
cluster_names_and_description=get_cluster_names_and_descriptions() # wczytujemy nazwy i opisy klastrów

predicted_cluster_id = predict_model(model, data=person_df)["Cluster"].values[0]
# # uruchamiamy funkcje predict_model, która zwraca nam id klastra, do którego należy osoba, której dane wprowadziliśmy
predicted_cluster_data = cluster_names_and_description[predicted_cluster_id]

st.header(f"Najbliżej ci do grupy: {predicted_cluster_data['name']}")
st.markdown(predicted_cluster_data["description"])
# # teraz możemy znaleźć osoby, które należą do tego samego klastra
some_cluster_df = all_df[all_df["Cluster"] == predicted_cluster_id]
st.metric("Podobnych osób w bazie", len(some_cluster_df)) # odejmujemy 1, bo osoba, której dane wprowadziliśmy też jest w tym klastrze

st.header("Osoby z grupy")
fig = px.histogram(some_cluster_df.sort_values("age"), x="age")
fig.update_layout(
    title="Rozkład wieku w grupie",
    xaxis_title="Wiek",
    yaxis_title="Liczba osób",
)
st.plotly_chart(fig)

fig = px.histogram(some_cluster_df, x="edu_level")
fig.update_layout(
    title="Rozkład wykształcenia w grupie",
    xaxis_title="Wykształcenie",
    yaxis_title="Liczba osób",
)
st.plotly_chart(fig)

fig = px.histogram(some_cluster_df, x="fav_animals")
fig.update_layout(
    title="Rozkład ulubionych zwierząt w grupie",
    xaxis_title="Ulubione zwierzęta",
    yaxis_title="Liczba osób",
)
st.plotly_chart(fig)

fig = px.histogram(some_cluster_df, x="fav_place")
fig.update_layout(
    title="Rozkład ulubionych miejsc w grupie",
    xaxis_title="Ulubione miejsce",
    yaxis_title="Liczba osób",
)
st.plotly_chart(fig)