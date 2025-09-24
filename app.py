# app.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

st.title("CORD-19 Data Explorer")
st.write("A simple Streamlit app to explore COVID-19 research papers.")

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("data/metadata.csv", low_memory=False)
    df['publish_time'] = pd.to_datetime(df['publish_time'], errors='coerce')
    df['year'] = df['publish_time'].dt.year
    df['abstract_word_count'] = df['abstract'].fillna("").apply(lambda x: len(x.split()))
    df = df.dropna(subset=['title'])
    return df

df = load_data()

# Sidebar controls
st.sidebar.header("Filters")
years = st.sidebar.slider("Select Year Range", int(df['year'].min()), int(df['year'].max()), (2020, 2021))
filtered_df = df[(df['year'] >= years[0]) & (df['year'] <= years[1])]

st.write(f"Showing {len(filtered_df)} papers between {years[0]} and {years[1]}")

# Publications by Year
st.subheader("Publications by Year")
year_counts = filtered_df['year'].value_counts().sort_index()
fig, ax = plt.subplots()
sns.barplot(x=year_counts.index, y=year_counts.values, ax=ax, color="skyblue")
ax.set_title("Publications by Year")
st.pyplot(fig)

# Top Journals
st.subheader("Top Journals")
top_journals = filtered_df['journal'].value_counts().head(10)
fig, ax = plt.subplots()
sns.barplot(y=top_journals.index, x=top_journals.values, ax=ax, color="lightgreen")
ax.set_title("Top 10 Journals")
st.pyplot(fig)

# Word Cloud
st.subheader("Word Cloud of Paper Titles")
titles = " ".join(filtered_df['title'].dropna().astype(str).tolist())
wordcloud = WordCloud(width=800, height=400, background_color="white").generate(titles)
fig, ax = plt.subplots(figsize=(10,6))
ax.imshow(wordcloud, interpolation="bilinear")
ax.axis("off")
st.pyplot(fig)

# Data Preview
st.subheader("Sample Data")
st.dataframe(filtered_df.head(20))
