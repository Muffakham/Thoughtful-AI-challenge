import numpy as np
import config
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class QA_Agent():
  def __init__(self):
    self.questions = config.QUESTIONS
    self.responses = config.QA_DATASET
    self.threshold = config.THRESHOLD
    self.tfidf_vectorizer, self.transformer_model = self.load_model()
    self.combined_question_embeddings = self.load_question_embeddings()

  def load_model(self):
    "Funciton to lazy load the TF-IDF and transformer models"
    
    try:
      tfidf_vectorizer = TfidfVectorizer()
    except Exception as e:
      logging.error(f"Error loading the TF-IDF vectorizer: {e}")
      tfidf_vectorizer = None
    try:
      sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
    except Exception as e:
      logging.error(f"Error loading the transformer model: {e}")
      sentence_model = None

    return [tfidf_vectorizer, sentence_model]


  def get_tfidf_embeddings(self, input, dataset: bool = False):
    "Function to get the TF-IDF embeddings of the input"
    try:
      if dataset:
        tfidf_matrix = self.tfidf_vectorizer.fit_transform(input)
      else:
        tfidf_matrix = self.tfidf_vectorizer.transform(input)
    except Exception as e:
      logging.error(f"Error getting the TF-IDF embeddings: {e}")
      tfidf_matrix = None
    return tfidf_matrix

  def get_transformer_embeddings(self, input):
    "Function to get the transformer embeddings of the input"
    try:
      sentence_embeddings = self.transformer_model.encode(input, convert_to_tensor=True)
    except Exception as e:
      logging.error(f"Error getting the transformer embeddings: {e}")
      sentence_embeddings = None
    return sentence_embeddings

  def load_question_embeddings(self):
    "Function to load the question embeddings"
    
    combined_question_embeddings = None
    try:
      
      question_tf_idf = self.get_tfidf_embeddings(self.questions, True)
      question_transformer = self.get_transformer_embeddings(self.questions)
      if question_transformer is None:
        combined_question_embeddings = question_tf_idf
      elif question_tf_idf is None:
        combined_question_embeddings = question_transformer
      elif question_tf_idf is not None and question_transformer is not None:
        scaled_tfidf_vector = question_tf_idf.toarray() * 0.4
        combined_question_embeddings = np.concatenate((scaled_tfidf_vector, question_transformer.cpu().detach().numpy()), axis=1)
      else:
        raise Exception("Error getting the question embeddings")

    except Exception as e:
      logging.error(f"Error loading the question embeddings: {e}")
      combined_question_embeddings = None
    
    return combined_question_embeddings

  def get_user_input_embeddings(self, user_input):
    """
    Funtion to get the combined tf-idf and transformer embeddings 
    for the user input
    """
    combined_user_input_embeddings = None
    try:
      
      user_input_tfidff = self.get_tfidf_embeddings([user_input])
      user_input_transformer = self.get_transformer_embeddings([user_input])
      if user_input_transformer is None:
        combined_user_input_embeddings = user_input_tfidff
      elif user_input_tfidff is None:
        combined_user_input_embeddings = user_input_transformer
      elif user_input_tfidff is not None and user_input_transformer is not None:
        scaled_tfidf_vector = user_input_tfidff.toarray() * 0.4
        combined_user_input_embeddings = np.concatenate((scaled_tfidf_vector, user_input_transformer.cpu().detach().numpy()), axis=1)
      else:
        raise Exception("Error getting the user input embeddings")
    
    except Exception as e:
      logging.error(f"Error getting the user input embeddings: {e}")
      combined_user_input_embeddings = None
    return combined_user_input_embeddings


  def calculate_answer(self, user_input_embeddings, question_embeddings):
    "Function to calculate the cosine similarity between the user input and the question embeddings"  
    response = None
    try:
      
      cosine_similarities = cosine_similarity(user_input_embeddings, question_embeddings)
      best_match_index = np.argmax(cosine_similarities)
      logging.info(f"Best match index: {best_match_index}")
      logging.info(f"Cosine similarities: {cosine_similarities}") 
      print(cosine_similarities)
      print(best_match_index)
      print(cosine_similarities[0][best_match_index])
      #print(config.THRESHOLD)
      if cosine_similarities[0][best_match_index] >= self.threshold:
          response = self.responses[self.questions[best_match_index]]
          print(response)
      else:
          response = config.FALLABCK_RESPONSE

    except Exception as e:
      
      logging.error(f"Error calculating the answer: {e}")
      response = None
    
    return response

  def get_response(self, user_input):
    "Function to get the response for the user input"
    try:

      user_input_embeddings = self.get_user_input_embeddings(user_input)
      output = self.calculate_answer(user_input_embeddings, self.combined_question_embeddings)
      if output is None:
        output = config.DEFAULT_RESPONSE
      
    except Exception as e:
      logging.error(f"Error getting the response: {e}") 
      output = config.DEFAULT_RESPONSE
    
    return output
