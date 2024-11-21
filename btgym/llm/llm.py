import os
# os.environ["OPENAI_API_KEY"]="sk-4vD6bVtv67XcfoVS8802AdF75888473296D604D707FbC9Bf"
# os.environ["OPENAI_BASE_URL"]= "https://gtapi.xiaoerchaoren.com:8932"

# from openai import OpenAI
import http.client
import json


DEFAULT_BASE_URL = "www.dwyu.top"
DEFAULT_API_KEY = "sk-CSHOawi329bdxl4nkaHWJDVfp8bj6pDFzM2vfjLFva0F7Msy"


class LLM():
    def __init__(self):
        self.base_url = DEFAULT_BASE_URL
        self.api_key = DEFAULT_API_KEY
        self.llm_model = "gpt-4o-mini"

    def request(self,message):
        conn = http.client.HTTPSConnection(self.base_url)
        payload = json.dumps({
            "model": self.llm_model,
            "messages": [
                {"role": "user", "content": message}
            ]
        })
        headers = {
            'Authorization': f' {self.api_key}',
            'Accept': 'application/json',
            'Content-Type': 'application/json'
        }
        conn.request("POST", "/v1/chat/completions", payload, headers)
        data = conn.getresponse().read()
        response = json.loads(data.decode("utf-8"))
        print(response)
        answer = response['choices'][0]['message']['content']
        return answer

if __name__ == '__main__':
    # llm = LLMGPT3()
    # embedding_models = llm.list_embedding_models()
    # print("Available embedding models:")
    # for model in embedding_models:
    #     print(model)

    # models = llm.list_models()
    # for model in models:
    #     print(model.id)

    # answer = llm.request(question="who are you,gpt?")
    # answer = llm.embedding(question="who are you,gpt?")
    # print(answer)
    #
    # llm = LLM()
    # answer = llm.request("who are you,gpt?")
    # print(answer)
    # print(llm.list_embedding_models())

    llm = LLM()
    llm.llm_model = "gpt-4-turbo"
    answer = llm.request("你是谁?你是claude吗，还是GPT?")
    print(answer)
    # while True:
    #     prompt = input("请输入你的问题:")
    #     answer = llm.request(prompt)
    #     print(answer)
