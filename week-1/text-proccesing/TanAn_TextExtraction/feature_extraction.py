import re
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, HashingVectorizer

def load_and_clean_data(file_path):
   df = pd.read_json(file_path)

   df['cleaned_content'] = df['content'].apply(clean_text)
   
   df = df[df['cleaned_content'].str.len() > 0].reset_index(drop=True)
   print(f"Số lượng bản ghi sau khi lọc: {len(df)}")
   
   return df

def clean_text(text):
   if isinstance(text, str):
       text = text.lower()
       text = re.sub(r'[^\w\s\u00C0-\u1EF9]', ' ', text)
       text = re.sub(r'\s+', ' ', text).strip()
       return text
   return ""

def extract_bow_features(df, max_features=1000, visualize=True):
   """
   Bag of Words sử dụng CountVectorizer
   """
   print("\n1. BAG OF WORDS (CountVectorizer)")
   count_vectorizer = CountVectorizer(max_features=max_features)
   bow_features = count_vectorizer.fit_transform(df['cleaned_content'])
   
   print(f"Kích thước ma trận BoW: {bow_features.shape}")
   print(f"Số từ vựng: {len(count_vectorizer.get_feature_names_out())}")
   print(f"sparsity: {1.0 - bow_features.nnz / (bow_features.shape[0] * bow_features.shape[1]):.4f}")
   
   word_counts = np.asarray(bow_features.sum(axis=0)).flatten()
   word_counts_dict = dict(zip(count_vectorizer.get_feature_names_out(), word_counts))
   top_words = sorted(word_counts_dict.items(), key=lambda x: x[1], reverse=True)[:20]
   
   print("\nTop 20 từ xuất hiện nhiều nhất:")
   for word, count in top_words:
       print(f"{word}: {count}")
   
   if visualize:
       plt.figure(figsize=(12, 6))
       words, counts = zip(*top_words)
       sns.barplot(x=list(counts), y=list(words))
       plt.title('Top 20 từ phổ biến nhất')
       plt.xlabel('Tần suất')
       plt.ylabel('Từ')
       plt.tight_layout()
       plt.savefig('top_words.png')
   
   return bow_features, count_vectorizer, word_counts_dict

def extract_tfidf_features(df, max_features=1000):
   """
   TF-IDF
   """
   print("\n2. TF-IDF (Term Frequency-Inverse Document Frequency)")
   tfidf_vectorizer = TfidfVectorizer(max_features=max_features)
   tfidf_features = tfidf_vectorizer.fit_transform(df['cleaned_content'])
   
   print(f"Kích thước ma trận TF-IDF: {tfidf_features.shape}")
   print(f"Độ thưa thớt (sparsity): {1.0 - tfidf_features.nnz / (tfidf_features.shape[0] * tfidf_features.shape[1]):.4f}")
   
   tfidf_means = np.asarray(tfidf_features.mean(axis=0)).flatten()
   tfidf_scores = dict(zip(tfidf_vectorizer.get_feature_names_out(), tfidf_means))
   top_tfidf = sorted(tfidf_scores.items(), key=lambda x: x[1], reverse=True)[:20]
   
   print("\nTop 20 từ có trọng số TF-IDF cao nhất:")
   for word, score in top_tfidf:
       print(f"{word}: {score:.4f}")
   
   return tfidf_features, tfidf_vectorizer, tfidf_scores

def extract_hash_features(df, n_features=1000):
   """
  HashingVectorizer
   """
   print("\n3. HASHING VECTORIZER")
   hashing_vectorizer = HashingVectorizer(n_features=n_features)
   hash_features = hashing_vectorizer.fit_transform(df['cleaned_content'])
   
   print(f"Kích thước ma trận hash: {hash_features.shape}")
   print(f"Độ thưa thớt (sparsity): {1.0 - hash_features.nnz / (hash_features.shape[0] * hash_features.shape[1]):.4f}")
   
   return hash_features, hashing_vectorizer

def analyze_topics(df, visualize=True):
   """
   Phân tích phân bố và từ đặc trưng theo chủ đề
   """
   if 'topic' not in df.columns:
       print("Không tìm thấy cột 'topic' trong dữ liệu")
       return None
   
   print("\n4. PHÂN TÍCH THEO CHỦ ĐỀ")
   topic_counts = df['topic'].value_counts()
   topic_percentage = topic_counts / len(df)
   
   print("Phân bố các chủ đề:")
   for topic, count in topic_counts.items():
       print(f"{topic}: {count} ({topic_percentage[topic]:.1%})")

   if visualize:
       plt.figure(figsize=(14, 7))
       topic_counts[:15].plot(kind='bar')
       plt.title('15 chủ đề phổ biến nhất')
       plt.xlabel('Chủ đề')
       plt.ylabel('Số lượng bài viết')
       plt.xticks(rotation=45, ha='right')
       plt.tight_layout()
       plt.savefig('topic.png')
   
   topic_keywords = {}
   print("\nCác từ đặc trưng theo chủ đề:")
   for topic in topic_counts.index[:5]: 
       topic_docs = df[df['topic'] == topic]['cleaned_content']
       topic_vectorizer = CountVectorizer(max_features=10)
       topic_vectors = topic_vectorizer.fit_transform(topic_docs)
       topic_word_counts = np.asarray(topic_vectors.sum(axis=0)).flatten()
       topic_word_dict = dict(zip(topic_vectorizer.get_feature_names_out(), topic_word_counts))
       top_topic_words = sorted(topic_word_dict.items(), key=lambda x: x[1], reverse=True)[:10]
       
       topic_keywords[topic] = top_topic_words
       
       print(f"\nChủ đề: {topic}")
       for word, count in top_topic_words:
           print(f"{word}: {count}")
   
   return {
       'topic_counts': topic_counts,
       'topic_percentage': topic_percentage,
       'topic_keywords': topic_keywords
   }

def main():
   df = load_and_clean_data('./VietnameseOnlineNewsDataset.txt')
   
   bow_features, count_vectorizer, word_counts = extract_bow_features(df)
   tfidf_features, tfidf_vectorizer, tfidf_scores = extract_tfidf_features(df)
   hash_features, hash_vectorizer = extract_hash_features(df)

   topic_analysis = analyze_topics(df)
   
   return {
       'dataframe': df,
       'bow_features': bow_features,
       'count_vectorizer': count_vectorizer,
       'word_counts': word_counts,
       'tfidf_features': tfidf_features,
       'tfidf_vectorizer': tfidf_vectorizer,
       'tfidf_scores': tfidf_scores,
       'hash_features': hash_features,
       'hash_vectorizer': hash_vectorizer,
       'topic_analysis': topic_analysis
   }

if __name__ == "__main__":
   results = main()