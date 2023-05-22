Python 3.11.3 (tags/v3.11.3:f3909b8, Apr  4 2023, 23:49:59) [MSC v.1934 64 bit (AMD64)] on win32
Type "help", "copyright", "credits" or "license()" for more information.
>>>import os
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

class Agent:
    def __init__(self):
        self.files = []
        self.documents = []
        self.default_answer = "I'm sorry, I don't have an answer to that question."

    def upload_files(self):
        num_files = int(input("How many files would you like to upload? "))
        for _ in range(num_files):
            file_path = input("Enter the file path: ")
            if os.path.exists(file_path):
                self.files.append(file_path)
                print(f"File '{file_path}' uploaded successfully.")
            else:
                print(f"File '{file_path}' does not exist.")

    def process_files(self):
        print("Processing files...")
        for file_path in self.files:
            with open(file_path, 'r') as file:
                content = file.read()
                self.documents.append(content)
        print("Files processed successfully.")

    def preprocess_text(self, text):
        # Tokenize the text
        tokens = word_tokenize(text.lower())

        # Remove stopwords
        stop_words = set(stopwords.words('english'))
        tokens = [token for token in tokens if token not in stop_words]

        # Lemmatize the tokens
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(token) for token in tokens]

        # Reconstruct the preprocessed text
        preprocessed_text = ' '.join(tokens)
        return preprocessed_text

    def generate_answer(self, question):
        # Preprocess the question
        preprocessed_question = self.preprocess_text(question)

        # Preprocess the documents
        preprocessed_documents = [self.preprocess_text(doc) for doc in self.documents]

        # Create the TF-IDF vectorizer
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(preprocessed_documents + [preprocessed_question])

        # Calculate cosine similarities between the question and the documents
        similarities = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])

        # Find the most similar document
        most_similar_index = similarities.argmax()

        # Generate the answer
        if similarities[0][most_similar_index] > 0:
            return self.documents[most_similar_index]
        else:
            return self.default_answer


# Create an instance of the agent
agent = Agent()

# Upload files
agent.upload_files()

# Process the uploaded files
agent.process_files()

# Interact with the agent
while True:
    user_question = input("Ask a question (or enter 'exit' to quit): ")
    if user_question.lower() == 'exit':
        break
    answer = agent.generate_answer(user_question)
    print(answer)
