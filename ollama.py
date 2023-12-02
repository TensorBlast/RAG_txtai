import ast
import requests
from pydantic import BaseModel, Field
from typing import List, Optional
from tqdm import tqdm 

class Parameters(BaseModel):
    num_keep : int = Field(5)
    seed : int = Field(0)
    num_predict : int = Field(128)
    top_k : int = Field(40)
    top_p : float = Field(0.9)
    tfs_z : float = Field(1)
    typical_p : float = Field(0.7)
    repeat_last_n : int = Field(64)
    temperature : float = Field(0.8)
    repeat_penalty : float = Field(1.1)
    presence_penalty : float = Field(1.5)
    frequency_penalty : float = Field(1.0)
    mirostat : float = Field(0)
    mirostat_tau : float = Field(5.0)
    mirostat_eta : float = Field(0.1)
    penalize_newline : bool = Field(True)
    stop : Optional[List] = Field(None)
    numa : bool = Field(False)
    num_ctx : int = Field(4096)
    num_batch : int = Field(2)
    num_gqa : int = Field(8)
    num_gpu : int = Field(1)
    main_gpu : int = Field(0)
    low_vram : bool = Field(False)
    f16_kv : bool = Field(True)
    logits_all : bool = Field(False)
    vocab_only : bool = Field(False)
    use_mmap : bool = Field(True)
    use_mlock : bool = Field(False)
    embedding_only : bool = Field(False)
    rope_frequency_base : float = Field(1.1)
    rope_frequency_scale : float = Field(0.8)
    num_thread : Optional[int] = Field(None)

class Ollama:
    def __init__(self, url = "http://localhost:11434/api"):
        self.url = url

    def list_models(self):
        resp = requests.get(self.url + "/tags")
        return [model['name'] for model in resp.json()['models']]
    
    def delete_model(self, model):
        return requests.delete(self.url + "/delete", json={"name": model})
    
    def pull_model(self, model):
        status = requests.post(self.url + "/pull", json={"name": model, "stream": True}, stream=True)
        print(f"Pulling {model}")
        prev_digest = None
        for line in status.iter_lines():
            response = ast.literal_eval(line.decode('utf-8'))
            if response['status'] == 'success':
                print(f"Successfully pulled {model}")
            digest = response['digest'] if 'digest' in response else None
            if digest and digest != prev_digest:
                print(f"\n{response['status']}:{digest}")
                prev_digest = digest
                pbar = tqdm(total=response['total'])
                if 'completed' in response:
                    pbar.update(response['completed']-pbar.n)
                    if response['completed'] == response['total']:
                        pbar.close()
            elif digest:
                if 'completed' in response:
                    pbar.update(response['completed']-pbar.n)
                    pbar.total = response['total']
                    pbar.refresh()
                    if response['completed'] == response['total']:
                        pbar.close()
                

    
    def generate(self, model, prompt, **kwargs):
        if kwargs:
            params = Parameters(**kwargs)
        else:
            params = Parameters()
        js = {"model": model, "prompt": prompt, "stream": False, "options": params.model_dump()}
        resp = requests.post(self.url + "/generate", json=js)
        return resp.json()['response']
    
    def rag_response(self, model, query, context):
        prompt = f"""
        Answer the following question using only the context below. Only include information specifically discussed.
        Question: {query}
        Context: {context}
        """
        return self.generate(model, prompt)


if __name__ == '__main__':
    ollama = Ollama()
    model = ollama.list_models()
    
    print(ollama.generate('mistral:7b-instruct-q8_0', "Who sang the song 'The Real Slim Shady'?", temperature=0.5, top_p=0.95, top_k=1000))
    