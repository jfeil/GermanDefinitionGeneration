import requests


class Model:
    Intel_Neural_Chat_7B = "intel-neural-chat-7b"
    Mixtral_8x7b = "mixtral-8x7b-instruct"
    Qwen_1_5_72B = "qwen1.5-72b-chat"
    Llama_3_70B = "meta-llama-3-70b-instruct"
    GPT_35 = "openai-gpt35"
    GPT_4 = "openai-gpt4"


def send_prompt(prompt: str, model: Model = Model.Llama_3_70B) -> str:
    endpoint = "https://chat-ai.academiccloud.de/chat-ai-backend"
    content = {
        "model": model,
        "messages":
            [
                {"role": "system", "content": "You are a helpful assistant"},
                {"role": "user", "content": "Test 123"},
                {"role": "assistant",
                 "content": "It seems like you're testing to see if I'm working properly! That's perfectly fine. I'm happy to report that I'm functioning as intended and ready to assist you with any questions or tasks you may have. Is there anything specific you'd like to chat about or ask? I'm all ears!"},
                {"role": "user", "content": "Test 123"},
                {"role": "assistant",
                 "content": "Your test is successful! How can I assist you with your scientific inquiries?"},
                {"role": "user", "content": "Test Test 123"}
            ],
        "temperature": 0.1
    }
    content = {
        "model": model,
        "messages":
            [
                {"role": "system", "content": " "},
                {"role": "user", "content": prompt},
            ],
        "temperature": 1
    }

    header = {"Content-Type": "application/json",
              "Cookie": "mod_auth_openidc_session=6ee85dde-56e1-49e3-a284-56ab3966fa3e"}

    resp = requests.post(endpoint, json=content, headers=header)
    return resp.content.decode()
