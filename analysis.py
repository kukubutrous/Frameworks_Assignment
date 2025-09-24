# analysis.py
"""
CORD-19 Metadata Analysis
-------------------------
This script loads, cleans, and analyzes the metadata.csv file
from the CORD-19 dataset. It also generates basic visualizations.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

# ============================
# Part 1: Load the dataset
# ============================
print("Loading dataset (this may take time for large files)...")
df = pd.read_csv("data/metadata.csv", low_memory=False)
print("✅ Loaded dataset with", df.shape[0], "rows and", df.shape[1], "columns")

# Display basic info
print("\n--- Data Info ---")
print(df.info())
print("\n--- Missing Values ---")
print(df.isnull().sum().head(15))  # check first 15 columns

# ============================
# Part 2: Data Cleaning
# ============================
print("\nCleaning data...")
# Convert publish_time to datetime
df['publish_time'] = pd.to_datetime(df['publish_time'], errors='coerce')
df['year'] = df['publish_time'].dt.year

# Create a new column: abstract word count
df['abstract_word_count'] = df['abstract'].fillna("").apply(lambda x: len(x.split()))

# Drop rows with no title
df = df.dropna(subset=['title'])
print("✅ Cleaned dataset with", df.shape[0], "rows remaining")

# ============================
# Part 3: Analysis + Visualizations
# ============================

# 1. Publications by year
year_counts = df['year'].value_counts().sort_index()
plt.figure(figsize=(8,5))
sns.barplot(x=year_counts.index, y=year_counts.values, color="skyblue")
plt.title("Publications by Year")
plt.xlabel("Year")
plt.ylabel("Number of Papers")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("publications_by_year.png")
plt.show()

# 2. Top journals
top_journals = df['journal'].value_counts().head(10)
plt.figure(figsize=(8,5))
sns.barplot(y=top_journals.index, x=top_journals.values, color="lightgreen")
plt.title("Top 10 Journals Publishing COVID-19 Research")
plt.xlabel("Number of Papers")
plt.ylabel("Journal")
plt.tight_layout()
plt.savefig("top_journals.png")
plt.show()

# 3. Word cloud of paper titles
titles = " ".join(df['title'].dropna().astype(str).tolist())
wordcloud = WordCloud(width=800, height=400, background_color="white").generate(titles)
plt.figure(figsize=(10,6))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.title("Word Cloud of Paper Titles")
plt.savefig("title_wordcloud.png")
plt.show()

# 4. Paper counts by source
source_counts = df['source_x'].value_counts().head(10)
plt.figure(figsize=(8,5))
sns.barplot(y=source_counts.index, x=source_counts.values, color="salmon")
plt.title("Top 10 Sources of Papers")
plt.xlabel("Number of Papers")
plt.ylabel("Source")
plt.tight_layout()
plt.savefig("sources.png")
plt.show()

print("\n✅ Analysis complete! Plots saved as PNG files.")
