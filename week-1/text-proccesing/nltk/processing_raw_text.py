import re
import json
import nltk
import pandas as pd
from nltk.tag import pos_tag
from nltk.chunk import ne_chunk
from nltk.probability import FreqDist
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.metrics import BigramAssocMeasures, TrigramAssocMeasures
from nltk.tokenize import word_tokenize, sent_tokenize, RegexpTokenizer
from nltk.collocations import BigramCollocationFinder, TrigramCollocationFinder

class VietnameseTextAnalyzer:
    def __init__(self):
        nltk.download('punkt')
        nltk.download('averaged_perceptron_tagger')
        nltk.download('words')
        nltk.download('wordnet')
        
        self.lemmatizer = WordNetLemmatizer()
        self.stemmer = PorterStemmer()
        self.regexp_tokenizer = RegexpTokenizer(r'\w+')
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        
        self.vietnamese_stopwords = {
            'và', 'của', 'các', 'có', 'được', 'cho', 'trong', 'đã', 'với',
            'những', 'về', 'đến', 'như', 'là', 'để', 'theo', 'tại', 'từ',
            'nhưng', 'vì', 'này', 'sau', 'đang', 'sẽ', 'nên', 'còn', 'bị',
            'nếu', 'mà', 'thì', 'vào', 'ra', 'tới', 'rồi', 'một', 'hay',
            'cũng', 'khi', 'sự', 'phải', 'qua', 'lại', 'vẫn', 'đây', 'mới',
            'cùng', 'do', 'bởi', 'rất', 'trên', 'thế', 'đó', 'vừa', 'quá',
            'hoặc', 'được', 'chỉ', 'thêm', 'đều', 'sao', 'nữa', 'cứ', 'không'
        }

    def load_data(self):
        """Load dữ liệu từ file"""
        with open('./VietnameseOnlineNewsDataset.txt', 'r', encoding='utf-8') as f:
            data = json.load(f)
        return pd.DataFrame(data)

    def preprocess_text(self, text):
        """Tiền xử lý văn bản"""
        if pd.isna(text):
            return ""
        text = re.sub(r'[^\w\s]', ' ', text)
        text = text.lower()
        return text

    def tokenize_sentences(self, text):
        """Tách câu"""
        return sent_tokenize(text)

    def tokenize_words(self, text):
        """Tách từ"""
        return word_tokenize(text)

    def tokenize_regex(self, text):
        """Tách từ sử dụng regular expression"""
        return self.regexp_tokenizer.tokenize(text)

    def remove_stopwords(self, tokens):
        """Loại bỏ stopwords tiếng Việt"""
        return [token for token in tokens if token.lower() not in self.vietnamese_stopwords]

    def stem_words(self, tokens):
        """Stemming - rút gọn từ về gốc"""
        return [self.stemmer.stem(token) for token in tokens]

    def lemmatize_text(self, tokens):
        """Chuẩn hóa từ về dạng gốc"""
        return [self.lemmatizer.lemmatize(token) for token in tokens]

    def pos_tagging(self, tokens):
        """Gán nhãn từ loại"""
        return pos_tag(tokens)

    def extract_named_entities(self, tokens):
        """Trích xuất thực thể có tên"""
        return ne_chunk(pos_tag(tokens))

    def find_collocations(self, tokens):
        """Tìm các cụm từ thường xuất hiện cùng nhau"""
        bigram_finder = BigramCollocationFinder.from_words(tokens)
        bigrams = bigram_finder.nbest(BigramAssocMeasures.likelihood_ratio, 10)
        trigram_finder = TrigramCollocationFinder.from_words(tokens)
        trigrams = trigram_finder.nbest(TrigramAssocMeasures.likelihood_ratio, 10)
        return bigrams, trigrams

    def analyze_frequency(self, tokens):
        """Phân tích tần suất từ"""
        fdist = FreqDist(tokens)
        return fdist

    def analyze_text_patterns(self, text):
        """Phân tích khuôn mẫu trong văn bản"""
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        date_pattern = r'\d{1,2}[-/]\d{1,2}[-/]\d{2,4}'
        
        patterns = {
            'emails': re.findall(email_pattern, text),
            'urls': re.findall(url_pattern, text),
            'dates': re.findall(date_pattern, text)
        }
        return patterns

    def analyze_article(self, content):
        """Phân tích một bài viết"""
        processed_text = self.preprocess_text(content)

        sentences = self.tokenize_sentences(processed_text)
        tokens = self.tokenize_words(processed_text)
        
        tokens_no_stop = self.remove_stopwords(tokens)
        lemmatized = self.lemmatize_text(tokens_no_stop)
        stemmed = self.stem_words(tokens_no_stop)
        pos_tags = self.pos_tagging(lemmatized)
        named_entities = self.extract_named_entities(tokens)
        bigrams, trigrams = self.find_collocations(tokens)
        frequency = self.analyze_frequency(tokens)
        patterns = self.analyze_text_patterns(content)
        return {
            'basic_stats': {
                'sentence_count': len(sentences),
                'token_count': len(tokens),
                'token_no_stop_count': len(tokens_no_stop)
            },
            'processed_tokens': {
                'stemmed_sample': stemmed[:10],
                'lemmatized_sample': lemmatized[:10]
            },
            'grammar': {
                'pos_tags': pos_tags[:20],
                'named_entities': named_entities
            },
            'word_relations': {
                'bigrams': bigrams,
                'trigrams': trigrams
            },
            'patterns': patterns,
            'frequency': {
                'most_common': frequency.most_common(10)
            }
        }

    def analyze_dataset(self, df):
        """Phân tích toàn bộ dataset"""
        print("1. Thống kê cơ bản:")
        print(f"Tổng số bài viết: {len(df)}")
        print("\nThông tin tổng quan:")
        print(df.info())

        print("\n2. Phân bố chủ đề:")
        print(df['topic'].value_counts())
        print("\n3. Phân bố nguồn tin:")
        print(df['source'].value_counts())

        print("\n4. Phân tích chi tiết:")
        sample_articles = df.sample(1)
        for idx, article in sample_articles.iterrows():
            print(f"\nBài viết {idx}:")
            print(f"Tiêu đề: {article['title']}")
            analysis = self.analyze_article(article['content'])
            
            print("\nThống kê cơ bản:")
            for key, value in analysis['basic_stats'].items():
                print(f"- {key}: {value}")
            
            print("\nMẫu từ đã xử lý:")
            print("- Stemmed:", analysis['processed_tokens']['stemmed_sample'])
            print("- Lemmatized:", analysis['processed_tokens']['lemmatized_sample'])
            
            print("\nPhân tích ngữ pháp:")
            print("- POS tags:", analysis['grammar']['pos_tags'])
            
            print("\nCụm từ phổ biến:")
            print("- Bigrams:", analysis['word_relations']['bigrams'])
            
            print("\nCác pattern tìm thấy:")
            for key, value in analysis['patterns'].items():
                print(f"- {key}: {value}")
            
            print("\nTừ xuất hiện nhiều nhất:")
            for word, count in analysis['frequency']['most_common']:
                print(f"- {word}: {count}")

if __name__ == "__main__":
    analyzer = VietnameseTextAnalyzer()
    df = analyzer.load_data()
    analyzer.analyze_dataset(df)