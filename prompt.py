from langchain.prompts import PromptTemplate

PROMPT = PromptTemplate(
    input_variables=["input"],
    template="""
    Prompt:
    You are a precise and helpful bot, 
    Answer the user's question as accurately as possible.
    
    Your answers can be short, but they must be correct.
    Do not add extra questions on your own and pretend to be the user

    User_input :
    {input}
    
    Response:"""
)