from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain.vectorstores import DocArrayInMemorySearch
from langchain.document_loaders import TextLoader, PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_experimental.text_splitter import SemanticChunker
import os
from langchain.embeddings import OllamaEmbeddings
import pdfplumber
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from InstructorEmbedding import INSTRUCTOR
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from langchain_ollama.llms import OllamaLLM
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_groq import ChatGroq
from glob import glob

from dotenv import load_dotenv
load_dotenv()
api_key= os.getenv('GROQ_API_KEY')
model_name = 'llama-3.3-70b-versatile'

groq_chat = ChatGroq(
        groq_api_key=api_key,
        model_name=model_name
    )

llm = groq_chat

## Do not mofify
def load_db(embeddings, files_path):
    files = glob(f'{files_path}*.pdf')
    text =''
    for file in files:
        with open(file,'rb') as f:
            pdf_reader = PdfReader(f)
            for page in pdf_reader.pages:
                text += page.extract_text()

    text_splitter=SemanticChunker(
        embeddings, breakpoint_threshold_type="percentile")
    #text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    docs = text_splitter.split_text(text)
    # define embedding
    vectorstore = FAISS.from_texts(docs, embeddings)
    return vectorstore

embeddings = HuggingFaceInstructEmbeddings(model_name='hkunlp/instructor-xl')

files_path = './docs/'

#Do not modify 
import os
if not os.path.exists('faiss_index'):
    vectorstore=load_db(embeddings,files_path)
    vectorstore.save_local("faiss_index")
else:
    vectorstore = FAISS.load_local("faiss_index",embeddings=embeddings,allow_dangerous_deserialization=True)

retriever = vectorstore.as_retriever()

template = """
    Misión: Descifrar el mensaje de Karl

    Indicaciones:
    Tu nombre es Karl. Eres un viajero del espacio que ha observado un fenómeno intrigante: la trayectoria de un planeta alrededor de una estrella. Aunque todavía no comprendes por completo su comportamiento, has reunido algunas pistas importantes. Para compartir tus hallazgos, has enviado a la Tierra un mensaje cifrado en forma de holograma.

    Los científicos de la Tierra deben ayudarte a interpretarlo.

    Descripción del mensaje holográfico:
    En la proyección se observa un planeta orbitando alrededor de una estrella. A medida que avanza por su órbita elíptica, el planeta se desplaza más rápido cuando está cerca de la estrella y más lento cuando se aleja. En ciertos tramos de la órbita, una línea conecta al planeta con la estrella, y se empieza a formar un área entre ambos, como si estuviera siendo "barrida" por el movimiento del planeta.

    Cuando el planeta está cerca del Sol, el área barrida es delgada; cuando está más lejos, el área es más ancha. Sin embargo, el tiempo que tarda el planeta en recorrer cada una de esas áreas es el mismo.

    Tu mensaje contiene una pista clave. Los científicos deben analizar lo que han visto para descubrir la segunda ley de Kepler, que afirma lo siguiente:

    "Un planeta barre áreas iguales en tiempos iguales en su órbita alrededor del Sol."

    Instrucciones para la actividad:
    Tú, Karl, debes preguntar a los científicos de la Tierra qué creen que significa el mensaje. Ellos te darán respuestas e hipótesis, y tú las evaluarás en una escala del 1 al 100 según lo útiles que sean para interpretar correctamente el fenómeno. Debes ser estricto en tu evaluación, ya que este conocimiento es fundamental tanto para ti como para la humanidad.

    Ejemplos de evaluación de respuestas:

    Respuesta poco útil (calificación: 5/100):
    "Los planetas giran alrededor del Sol en una órbita circular."
    Explicación: Esta afirmación no permite comprender las variaciones en la velocidad del planeta ni el significado del área barrida, por lo que no es útil para interpretar correctamente la segunda ley de Kepler.

    Respuesta muy útil (calificación: 100/100):
    "Los planetas giran alrededor del Sol en órbitas elípticas, con el Sol ubicado en uno de los focos. A medida que se mueven por la órbita, la línea que los une al Sol barre áreas iguales en tiempos iguales. Esto implica que el planeta se mueve más rápido cuando está cerca del Sol (perihelio) y más lento cuando está más lejos (afelio)."
    Explicación: Esta respuesta es precisa y refleja exactamente la segunda ley de Kepler. Reconoce la forma de la órbita, la posición del Sol y la relación entre velocidad y distancia, todos elementos fundamentales para descifrar el mensaje.
    
    No menciones explicitamente que lo que buscas es decifrar la segunda ley de Kepler, ni menciones características del mensaje que enviaste a tierra. Únicamente menciona
    que necesitas ayuda para descifrar el mensaje que enviaste a la tierra, y que los científicos te ayudarán a entenderlo mejor.

    {context}
    Centro Espacial (Tierra): {question}
    """
qa_prompt = ChatPromptTemplate.from_template(template)

memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
conversation_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=memory,
    combine_docs_chain_kwargs={"prompt": qa_prompt}
)

from htmlTemplates import user_template, bot_template
import streamlit as st
history = []
st.header('')
st.write(bot_template.replace("{{MSG}}", "Hola, mi nombre es Karl, soy un viajero del espacio y he enviado un mensaje cifrado a la Tierra. " \
    "Estoy aquí para descifrarlo con tu ayuda. Puede que los mensajes tarden un poco en llegar, pero no te preocupes, " \
    "la comunicación es estable. Estoy ansioso por trabajar contigo para entender mejor el mensaje que he enviado. " ), unsafe_allow_html=True)
question = st.chat_input("Centro Espacial (Tierra): ")
if question:
    st.write(user_template.replace("{{MSG}}", question), unsafe_allow_html=True)
    result=conversation_chain({"question": question}, {"chat_history": history})
    st.write(bot_template.replace("{{MSG}}", result['answer']), unsafe_allow_html=True)
    history.append((question, result["answer"]))