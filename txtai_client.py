import requests

class txtaiClient:
    def __init__(self, url = "http://localhost:8000"):
        self.url = url
        self.headers = {"Content-Type": "application/json"}

    def embeddings(self, text):
        resp = requests.get(self.url + "/transform"+"?text="+text)
        return resp.json()
    
    def batch_embeddings(self, text: list, batch_size=1000):
        batch = []
        items = 0
        embeddings = []
        for l in text:
            batch.append(l)
            if len(batch) >= batch_size:
                resp = requests.post(self.url +"/batchtransform", headers=self.headers, json=batch)
                embeddings.extend(resp.json())
                items += len(embeddings)
                batch = []
        if len(batch) > 0:
            resp = requests.post(self.url +"/batchtransform", headers=self.headers, json=batch)
            embeddings.extend(resp.json())
            items += len(embeddings)
        print(f"Calculated {items} embeddings")
        return embeddings
    
    def add(self, batch: list|str):
        if isinstance(batch, str):
            batch = [{"text": batch}]
        elif isinstance(batch, list):
            batch = [{"text": item} for item in batch]
        resp = requests.post(self.url + "/add", headers=self.headers, json=batch, timeout=120)
        return resp
    
    def add_text(self, text: list, batch_size=1000):
        batch = []
        items = 0
        for l in text:
            batch.append(l)
            if len(batch) >= batch_size:
                resp = self.add(batch)
                if resp.status_code != 200:
                    print(resp.text)
                items += len(batch)
                batch = []
        if len(batch) > 0:
            self.add(batch)
            items += len(batch)
        print(f"Added {items} items")

    def add_file(self, file_path: str, batch_size=1000):
        with open(file_path) as f:
            self.add_text(f.readlines(), batch_size=batch_size)

    def index(self):
        print('Indexing...')
        resp = requests.get(self.url+"/index")
        if resp.status_code != 200:
            print(resp.text)
            raise Exception("Indexing failed")

    def search(self, query: str):
        resp = requests.get(self.url+"/search"+"?query="+query)
        return resp.json()
    
    def search_batch(self, query: list, limit: int=10, weights:int = 0, index: str=None):
        resp = requests.post(self.url+"/batchsearch", headers=self.headers, json={"queries": query, "limit": limit, "weights": weights, "index": index})
        return resp.json()
    
    def count(self):
        resp = requests.get(self.url+"/count")
        return resp.text
    
    def similarity(self, query: str, texts: list[str]):
        resp = requests.post(self.url+"/similarity", headers=self.headers, json={"query": query, "texts": texts})
        return resp.json()
    
    def batchsimilarity(self, queries: list[str], texts: list[str]):
        resp = requests.post(self.url+"/batchsimilarity", headers=self.headers, json={"queries": queries, "texts": texts})
        return resp.json()
    
    #Runs an upsert operation on previously batched (using textai_client.add/add_text/add_file) data
    def upsert(self):
        resp = requests.get(self.url+"/upsert")
        if resp.status_code != 200:
            print(resp.text)
            raise Exception("Upsert failed")
        
    def delete(self, idlist: list[str]):
        resp = requests.post(self.url+"/delete", headers=self.headers, json=idlist)
        if resp.status_code != 200:
            print(resp.text)
            raise Exception("Delete failed")
        else:
            return resp.json()

if __name__ == '__main__':
    client = txtaiClient()
    
    from datasets import load_dataset
    # data = load_dataset("ag_news", split="train")

    # client.add_text(data["text"], batch_size=4096)
    # client.index()

    print("Number of items in Index: ", client.count())
    # x=client.search_batch([input('Enter query: ')], limit=50)[0]
    # print(f"Search results: {x}")
    # print(f"Number of results: {len(x)}")
    # embed = client.batch_embeddings(["hello", "world", "what's up", "how are you doing?"])
    # print(embed)
    # print(f"Number of embeddings: {len(embed)}")
    # print(client.similarity("hello", ["hello", "world", "what's up", "how are you doing?"]))
    # print(client.batchsimilarity(["hello", "world", "what's up", "how are you doing?"], ["hello", "world", "what's up", "how are you doing?"]))
    # print("Deleting few items: ", client.delete(['103268','82356','108493']))
    client.index()