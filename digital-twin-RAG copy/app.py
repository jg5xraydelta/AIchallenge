import gradio as gr

def respond(message,history):
    response = f"You said: {message}\
        \nAnd I say I love learning AI Engineering with SuperDataScience!"
    return response

gr.ChatInterface(respond).launch()