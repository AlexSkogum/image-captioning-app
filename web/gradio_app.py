import requests
import gradio as gr


API_URL = 'http://127.0.0.1:8000/caption'


def predict(image):
    if image is None:
        return 'No image', None
    # Gradio returns PIL Image; send bytes
    from io import BytesIO
    buf = BytesIO()
    image.save(buf, format='PNG')
    buf.seek(0)
    files = {'file': ('img.png', buf.getvalue(), 'image/png')}
    data = {'beam_size': '3'}
    r = requests.post(API_URL, files=files, data=data)
    if r.status_code != 200:
        return f'API error: {r.status_code} {r.text}', None
    j = r.json()
    captions = j.get('captions', [])
    att_b64 = j.get('attention_base64')
    return '\n'.join(captions), att_b64


with gr.Blocks() as demo:
    gr.Markdown('# Image Captioning — Demo')
    with gr.Row():
        inp = gr.Image(type='pil')
        out = gr.Textbox(label='Captions')
    btn = gr.Button('Generate')
    btn.click(fn=predict, inputs=inp, outputs=[out, gr.Image(label='Attention')])

if __name__ == '__main__':
    demo.launch(server_name='0.0.0.0', server_port=7860)
