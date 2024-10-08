# Thoughtful AI Customer Support Chatbot

This project implements a basic customer support chatbot using Streamlit and various NLP techniques to answer common questions about Thoughtful AI and its agents.

## Project Structure

The project is structured as follows:

- **config.py:** Contains configuration variables, such as the threshold for cosine similarity, a list of pre-defined questions, and a dictionary mapping questions to their corresponding answers.
- **llm_agent.py:** Contains the core logic of the QA agent.
    - **QA_Agent class:**
        - Loads and initializes the necessary models for embedding generation (TF-IDF and Sentence Transformer).
        - Generates embeddings for both questions and user input.
        - Calculates cosine similarity between user input and questions.
        - Provides a response based on the similarity threshold and the matching question.
- **app.py:** Contains the Streamlit application logic.
    - Creates a simple user interface with a text input field for questions.
    - Processes user input by calling the QA agent.
    - Displays the response from the QA agent.

## Dependencies

The project requires the following Python libraries:

- numpy
- scikit-learn
- sentence-transformers
- streamlit

You can install them using the requirements.txt file:

