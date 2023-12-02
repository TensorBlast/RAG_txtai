from txtai_client import txtaiClient
from ollama import Ollama

def rag(text, ollamaClient, llm):
    client = txtaiClient()
    context = " ".join([ x['text'] for x in client.search(text, limit=3)])
    llm_response = ollamaClient.rag_response(llm, text, context)
    return llm_response


if __name__ == '__main__':
    ollamaClient = Ollama()
    model = 'mistral:7b-instruct-q8_0'
    query = "What is the name of some of the richest men in the world?"
    print(rag(query, ollamaClient, model))

    query2 = input("Enter your query: ")
    print('Answer: ', rag(query2, ollamaClient, model))