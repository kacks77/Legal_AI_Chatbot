import streamlit as st
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.schema import HumanMessage, AIMessage
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_google_genai import ChatGoogleGenerativeAI

# Define the AI legal assistant class
class LegalAI:
    def __init__(self, api_key):
        self.api_key = api_key
        self.model = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash", google_api_key=self.api_key
        )
        self.msgs = StreamlitChatMessageHistory(key="legal_case_history")

    def process_legal_case(self, case_text):
        if not case_text.strip():
            raise ValueError("Legal case text cannot be empty.")
        
        self.msgs.add_message(HumanMessage(content=case_text))
        messages = [
            {"type": "system", "content": "You are a legal assistant."},
            {"type": "human", "content": case_text},
        ]
        try:
            response = self.model.invoke(messages)
            if isinstance(response, dict) and "content" in response:
                ai_content = response["content"]
            elif hasattr(response, "content"):
                ai_content = response.content
            else:
                raise TypeError("Unexpected response type: {}".format(type(response)))
            
            self.msgs.add_message(AIMessage(content=ai_content))
            return ai_content
        except Exception as e:
            st.error(f"Error while processing legal case: {e}")
            return ""

    def answer_question(self, user_query):
        if not user_query.strip():
            st.error("Query cannot be empty.")
            return ""

        self.msgs.add_message(HumanMessage(content=user_query))
        messages = [
            {"type": "system", "content": "You are a legal assistant."},
            *[
                {"type": msg.type, "content": msg.content}
                for msg in self.msgs.messages
            ],
        ]
        try:
            response = self.model.invoke(messages)
            if isinstance(response, dict) and "content" in response:
                ai_content = response["content"]
            elif hasattr(response, "content"):
                ai_content = response.content
            else:
                raise TypeError("Unexpected response type: {}".format(type(response)))
            
            self.msgs.add_message(AIMessage(content=ai_content))
            return ai_content
        except Exception as e:
            st.error(f"Error while answering question: {e}")
            return ""

# Streamlit UI setup
st.set_page_config(page_title="Legal AI Assistant", page_icon="⚖️")
st.title("Legal AI Assistant")
st.write("Welcome! This chatbot helps you analyze and interact with legal cases.")

def get_api_key():
    try:
        return st.secrets["google_api_key"]
    except KeyError:
        st.error("Google API key is missing from secrets!")
        return None
    

api_key = get_api_key()


if api_key:
    legal_ai = LegalAI(api_key)
    st.write("You can now interact with the legal AI assistant.")
    
    case_text = st.text_area("Paste the legal case text here:")
    if case_text:
        case_summary = legal_ai.process_legal_case(case_text)
        if case_summary:
            st.write(f"Case Summary: {case_summary}")
    
    user_query = st.text_input("Ask a legal question:")
    if user_query:
        response = legal_ai.answer_question(user_query)
        if response:
            st.write(f"Answer: {response}")
else:
    st.warning("Please set up the API key in your Streamlit secrets.")
