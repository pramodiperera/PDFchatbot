from dotenv import load_dotenv
import os
import streamlit as st
from PyPDF2 import PdfReader
from langchain.llms import OpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback

def main():
    load_dotenv()
    #print('hello')
    #print(os.getenv("OPENAI_API_KEY"))

    st.set_page_config(page_title="Ask your PDF")
    st.header("Ask your PDF")
    pdf = st.file_uploader("Upload your PDF", type="pdf")
    
    # extract text
    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        #st.write(text)

        # split into chunks
        text_splitter = CharacterTextSplitter(
            separator = "\n",
            chunk_size = 500,
            chunk_overlap = 200,
            length_function = len
        )

        chunks = text_splitter.split_text(text)
        #st.write(chunks)

        # embedding
        embedding = OpenAIEmbeddings()
        knowledge_base = FAISS.from_texts(chunks, embedding)

        user_question = st.text_input("Ask a question about your PDF:")
        if user_question:
            docs = knowledge_base.similarity_search(user_question)
            #st.write(docs)

            llm = OpenAI()
            chain = load_qa_chain(llm, chain_type="stuff")
            with get_openai_callback() as cb:
                response = chain.run(input_documents = docs, question = user_question)
                print(cb)

            st.write(response)

if __name__ == '__main__':
    main()
