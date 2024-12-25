import openai
from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import HumanMessage, AIMessage
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_google_genai import ChatGoogleGenerativeAI
import streamlit as st
import json
from langdetect import detect
from PyPDF2 import PdfReader
from docx import Document
from deep_translator import GoogleTranslator
import re

# Define the AI legal assistant class
class LegalAI:
    def __init__(self, api_key):
        self.api_key = api_key
        self.model = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash", google_api_key=self.api_key, convert_system_message_to_human=True
        )
        self.msgs = StreamlitChatMessageHistory(key="legal_case_history")

    def process_legal_case(self, case_text):
        self.msgs.add_message(HumanMessage(content=case_text))
        messages = [
            {"type": "human", "content": "You are a legal assistant."},
            {"type": "human", "content": case_text},
        ]
        response = self.model.invoke(messages)

        if isinstance(response, dict) and "content" in response:
            ai_content = response["content"]
        elif hasattr(response, "content"):
            ai_content = response.content
        else:
            raise TypeError("Unexpected response type: {}".format(type(response)))

        self.msgs.add_message(AIMessage(content=ai_content))
        return ai_content

    def answer_question(self, user_query):
        self.msgs.add_message(HumanMessage(content=user_query))
        messages = [
            {"type": "human", "content": "You are a legal assistant."},
            *[
                {"type": msg.type, "content": msg.content}
                for msg in self.msgs.messages
            ],
        ]
        response = self.model.invoke(messages)

        if isinstance(response, dict) and "content" in response:
            ai_content = response["content"]
        elif hasattr(response, "content"):
            ai_content = response.content
        else:
            raise TypeError("Unexpected response type: {}".format(type(response)))

        self.msgs.add_message(AIMessage(content=ai_content))
        return ai_content


# Set up Streamlit UI
st.set_page_config(page_title="Legal AI Assistant", page_icon="⚖️")
st.title("Legal AI Assistant")
st.write("Welcome! This chatbot helps you analyze and interact with legal cases.")

# Get the API key
def get_api_key():
    if "api_key" not in st.session_state:
        st.session_state["api_key"] = ""
    return st.text_input("Enter your API key", type="password")

api_key = get_api_key()

# If API key is provided, start the assistant
if api_key:
    legal_ai = LegalAI(api_key)
    st.write("You can now interact with the legal AI assistant.")

    # Upload and process legal documents
    uploaded_doc = st.file_uploader("Upload a Legal Document (PDF or DOCX)", type=["pdf", "docx"])
    
    case_text = ""
    if uploaded_doc:
        if uploaded_doc.type == "application/pdf":
            reader = PdfReader(uploaded_doc)
            case_text = " ".join(page.extract_text() for page in reader.pages)
        elif uploaded_doc.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            doc = Document(uploaded_doc)
            case_text = " ".join([p.text for p in doc.paragraphs])
        st.text_area("Extracted Case Text", case_text)
    else:
        case_text = st.text_area("Paste the legal case text here:")

    # Ensure case_text is not empty
    if case_text.strip():
        # Language detection and translation
        language = st.selectbox("Select Language", ["English", "Spanish", "French", "German", "Auto-detect"])
        case_text_language = detect(case_text) if language == "Auto-detect" else language.lower()

        # Supported language codes for GoogleTranslator
        supported_languages = {
            'english': 'en', 'spanish': 'es', 'french': 'fr', 'german': 'de',
            'auto': 'auto'  # For auto-detection
        }

        # Validate selected language
        case_text_language_code = supported_languages.get(case_text_language)
        if not case_text_language_code:
            st.error(f"Selected language '{case_text_language}' is not supported.")
        else:
            # Function to handle text translation with chunking if needed
            def safe_translate(text, source_language, target_language="en", max_length=5000):
                try:
                    if len(text) > max_length:
                        chunks = [text[i:i + max_length] for i in range(0, len(text), max_length)]
                        translated_chunks = []
                        for chunk in chunks:
                            translated_chunk = GoogleTranslator(
                                source=source_language, target=target_language
                            ).translate(chunk)
                            translated_chunks.append(translated_chunk)
                        return " ".join(translated_chunks)
                    else:
                        return GoogleTranslator(source=source_language, target=target_language).translate(text)
                except Exception as e:
                    st.error(f"Translation error: {e}")
                    return text

            # Translation logic
            if case_text_language_code != "en":
                case_text_translated = safe_translate(
                    case_text, source_language=case_text_language_code, target_language="en"
                )
                st.write(f"Translated Case Text: {case_text_translated}")
                case_summary = legal_ai.process_legal_case(case_text_translated)
                case_summary_translated = safe_translate(
                    case_summary, source_language="en", target_language=case_text_language_code
                )
                st.write(f"Case Summary (Translated): {case_summary_translated}")
            else:
                summarization_level = st.selectbox("Summarization Level", ["Brief", "Detailed"])
                case_summary = legal_ai.process_legal_case(case_text + f" Summarize in a {summarization_level.lower()} way.")
                st.write(f"Case Summary ({summarization_level}): {case_summary}")

            # Find legal citations
            citations = re.findall(r"\b\d+\s+[A-Z]+\.\s+\d+\b", case_text)
            if citations:
                st.write("Found Legal Citations:")
                for citation in citations:
                    st.markdown(f"- [{citation}](https://scholar.google.com/scholar?q={citation})")
            else:
                st.write("No legal citations found.")

            # User's legal query
            user_query = st.text_input("Ask a legal question:")
            if user_query:
                response = legal_ai.answer_question(user_query)
                st.write(f"Answer: {response}")

            # Save and download chat history
            if st.button("Download Chat History"):
                history = [{"type": msg.type, "content": msg.content} for msg in legal_ai.msgs.messages]
                history_json = json.dumps(history, indent=4)
                st.download_button(
                    label="Download Chat History",
                    data=history_json,
                    file_name="chat_history.json",
                    mime="application/json"
                )

            # Upload chat history
            uploaded_history_file = st.file_uploader("Upload Chat History", type=["json"])
            if uploaded_history_file:
                try:
                    uploaded_history = json.load(uploaded_history_file)
                    legal_ai.msgs.clear()
                    for msg in uploaded_history:
                        if msg["type"] == "human":
                            legal_ai.msgs.add_message(HumanMessage(content=msg["content"]))
                        elif msg["type"] == "ai":
                            legal_ai.msgs.add_message(AIMessage(content=msg["content"]))
                    st.success("Chat history loaded successfully!")
                except Exception as e:
                    st.error(f"Failed to load chat history: {e}")

            # Feedback system
            feedback = st.radio("Was this response helpful?", ["Yes", "No"], key=f"feedback_{user_query}")
            if feedback == "No":
                st.text_area("How can we improve?", key=f"feedback_comment_{user_query}")

else:
    st.warning("Please enter your API key.")