import numpy as np
class TFIDFVectorizer:
    def __init__(self, max_features=5000):
        self.max_features = max_features
        self.vocabulary = {}    # word -> index
        self.idf = {}           # word -> idf score

    def fit(self, emails):
        freq = {}
        doc_freq = {}
        processed_emails = []
        
        for email in emails:
            # Preprocessing: lowercase + remove punctuation
            text = email.lower()
            text = ''.join(c if c.isalnum() or c.isspace() else ' ' for c in text)
            words = text.split()
            processed_emails.append(words)
            
            # For TF (overall corpus frequency)
            for word in words:
                freq[word] = freq.get(word, 0) + 1
                
            # For IDF (document frequency)
            unique_words = set(words)
            for word in unique_words:
                doc_freq[word] = doc_freq.get(word, 0) + 1
                
        sorted_words = sorted(freq, key=lambda word: freq[word], reverse=True)
        top_words = sorted_words[:self.max_features]
        
        for index, word in enumerate(top_words):
            self.vocabulary[word] = index
            self.idf[word] = np.log(len(emails) / doc_freq[word])
    
    def IDF(self, emails, word):
        # Deprecated: Now calculated efficiently in fit()
        pass
    
    def TF(self, email, word):
        # Deprecated: Now calculated efficiently in transform()
        pass

    def transform(self, emails):
        matrix = []
        
        for email in emails:
            # Preprocessing: lowercase + remove punctuation
            text = email.lower()
            text = ''.join(c if c.isalnum() or c.isspace() else ' ' for c in text)
            words = text.split()
            total_words = len(words)
            
            # Count term frequencies for this email
            word_counts = {}
            for word in words:
                word_counts[word] = word_counts.get(word, 0) + 1
                
            vector = np.zeros(len(self.vocabulary))
            # Only iterate through unique words in this email
            for word, count in word_counts.items():
                if word in self.idf:
                    tf = count / total_words if total_words > 0 else 0
                    vector[self.vocabulary[word]] = tf * self.idf[word]
                    
            matrix.append(vector)
        return np.array(matrix)

    def fit_transform(self, emails):
        self.fit(emails)
        return self.transform(emails)
    

