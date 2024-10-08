import streamlit as st
import llm_agent
from llm_agent import QA_Agent
import config
import logging

# Initialize logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize QA agent
qa_agent = QA_Agent()

# Streamlit app layout
st.title("Thoughtful AI Customer Support")
st.write("I'm here to help you with your questions about Thoughtful AI and its agents!")

# Input from user
user_input = st.text_input("Ask me a question:")

# Provide default response if no input is given
if user_input == "":
    response = config.DEFAULT_RESPONSE
else:
    try:        
        
        response = qa_agent.get_response(user_input)
        logging.info(f"User query: {user_input} | Response: {response}")

    except Exception as e:
        response = f"An error occurred while processing your request: {e}"

# Display the response
st.write(response)
