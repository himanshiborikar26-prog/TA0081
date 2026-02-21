from langchain_core.prompts import ChatPromptTemplate
prompt = ChatPromptTemplate.from_template(
    """Answer the question based only on the context below:

    Context:
    {context}

    Question:
    {question}
    """
)