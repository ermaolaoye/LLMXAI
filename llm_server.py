# %% AIHub API
import base64
import os
import requests

class LLMServer:
    PERSONALITY_DEFAULT = "You are a helpful AI assistant."
    PERSONALITY_COMMAND_ONLY = "You can only output commands, no explanations."
    PERSONALITY_JSON_ONLY = "You can only output JSON, no explanations."
    PERSONALITY_YAML_ONLY = "You can only output YAML, no explanations."

    MODEL_LLAMA3_70B = "llama3.3:latest" # 70B llama3.3 model
    MODEL_LLAMA3_3B = "llama3.2:latest" # 3B llama3.2 model

    def __init__(self, token=None):
        self.url = "https://ai.create.kcl.ac.uk/"
        self.token = token
        self.headers = {
            "Authorization": f"Bearer {self.token}"
        }
        self.set_system(LLMServer.PERSONALITY_DEFAULT)
        self.set_model(LLMServer.MODEL_LLAMA3_70B) # Default model

    def set_system(self, personality):
        self.personality = personality
        self.chat_history = [{"role": "system", "content": f"System personality set to {personality}"}]

    def set_model(self, model):
        if model != LLMServer.MODEL_LLAMA3_70B and model != LLMServer.MODEL_LLAMA3_3B:
            raise ValueError("Model not supported")

        self.model = model

    def ask(self, prompt, model=""):
        if model != "":
            self.set_model(model)
        data = {
            "model": self.model,
            "prompt": prompt,
            "stream": False
        }

        if self.personality != LLMServer.PERSONALITY_DEFAULT:
            data["system"] = self.personality

        url = self.url + "ollama/api/generate"
        response = requests.post(url, json=data, headers=self.headers)
        print(response.text)
        response = response.json()
        if "response" in response:
            return response["response"]
        if "detail" in response:
            raise Exception(response["detail"])
        raise Exception("Unknown error")

    def chat(self, prompt, model=""):
        if model != "":
            self.set_model(model)

        message = {"role": "user", "content": prompt}

        self.chat_history.append(message)

        data = {
            "model": self.model,
            "stream": True,
            "messages": self.chat_history
        }

        url = self.url + "ollama/api/chat"

        response = requests.post(url, json=data, headers=self.headers)
        response = response.json()

        if "messages" in response:
            message = response["messages"]
            self.chat_history.append(message)
            return response["content"]
        if "detail" in response:
            raise Exception(response["detail"])
        raise Exception("Unknown error")

llm = LLMServer(token="sk-763f51916aaa4a548cbb4bf365fb9e81")

response = llm.ask("What is the capital of France?")
print(response)
