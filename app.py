import os
import threading
import time
import subprocess

OLLAMA = os.path.expanduser("~/ollama")

if not os.path.exists(OLLAMA):
    subprocess.run("curl -L https://ollama.com/download/ollama-linux-amd64 -o ~/ollama", shell=True)
    os.chmod(OLLAMA, 0o755)

def ollama_service_thread():
    subprocess.run("~/ollama serve", shell=True)

OLLAMA_SERVICE_THREAD = threading.Thread(target=ollama_service_thread)
OLLAMA_SERVICE_THREAD.start()

print("Giving ollama serve a moment")
time.sleep(10)

# Modify the model to what you want
model = "gemma2"

subprocess.run(f"~/ollama pull {model}", shell=True)


import copy
import gradio as gr
from ollama import Client
client = Client(host='http://localhost:11434', timeout=120)

HF_TOKEN = os.environ.get("HF_TOKEN", None)
MODEL_ID = os.environ.get("MODEL_ID", "google/gemma-2-9b-it")
MODEL_NAME = MODEL_ID.split("/")[-1]

TITLE = "<h1><center>ollama-Chat</center></h1>"

DESCRIPTION = f"""
<h3>MODEL: <a href="https://hf.co/{MODEL_ID}">{MODEL_NAME}</a></h3>
<center>
<p>Feel free to test models with ollama.
<br>
Easy to modify and running models you want.
</p>
</center>
"""

CSS = """
.duplicate-button {
    margin: auto !important;
    color: white !important;
    background: black !important;
    border-radius: 100vh !important;
}
h3 {
    text-align: center;
}
"""


def stream_chat(message: str, history: list, temperature: float, max_new_tokens: int, top_p: float, top_k: int, penalty: float):
    
    conversation = []
    for prompt, answer in history:
        conversation.extend([
            {"role": "user", "content": prompt}, 
            {"role": "assistant", "content": answer},
        ])
    conversation.append({"role": "user", "content": message})

    print(f"Conversation is -\n{conversation}")

    response = client.chat(
        model=model,
        messages=conversation,
        stream=True,
        options={
            'num_predict': max_new_tokens,
            'temperature': temperature,
            'top_p': top_p,
            'top_k': top_k,
            'repeat_penalty': penalty,
            'low_vram': True,
        },
    )

    buffer = ""
    for chunk in response:
        buffer += chunk["message"]["content"]
        yield buffer



chatbot = gr.Chatbot(height=600)

with gr.Blocks(css=CSS, theme="soft") as demo:
    gr.HTML(TITLE)
    gr.HTML(DESCRIPTION)
    gr.DuplicateButton(value="Duplicate Space for private use", elem_classes="duplicate-button")
    gr.ChatInterface(
        fn=stream_chat,
        chatbot=chatbot,
        fill_height=True,
        additional_inputs_accordion=gr.Accordion(label="⚙️ Parameters", open=False, render=False),
        additional_inputs=[
            gr.Slider(
                minimum=0,
                maximum=1,
                step=0.1,
                value=0.8,
                label="Temperature",
                render=False,
            ),
            gr.Slider(
                minimum=128,
                maximum=2048,
                step=1,
                value=1024,
                label="Max New Tokens",
                render=False,
            ),
            gr.Slider(
                minimum=0.0,
                maximum=1.0,
                step=0.1,
                value=0.8,
                label="top_p",
                render=False,
            ),
            gr.Slider(
                minimum=1,
                maximum=20,
                step=1,
                value=20,
                label="top_k",
                render=False,
            ),
            gr.Slider(
                minimum=0.0,
                maximum=2.0,
                step=0.1,
                value=1.0,
                label="Repetition penalty",
                render=False,
            ),
        ],
        examples=[
            ["Help me study vocabulary: write a sentence for me to fill in the blank, and I'll try to pick the correct option."],
            ["What are 5 creative things I could do with my kids' art? I don't want to throw them away, but it's also so much clutter."],
            ["Tell me a random fun fact about the Roman Empire."],
            ["Show me a code snippet of a website's sticky header in CSS and JavaScript."],
        ],
        cache_examples=False,
    )


if __name__ == "__main__":
    demo.launch()