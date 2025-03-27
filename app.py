from flask import Flask, render_template, request, send_file
from diffusers import StableDiffusionXLPipeline
import torch


app = Flask(__name__)

# Load the model once
model_id = "stabilityai/stable-diffusion-xl-base-1.0"
pipe = StableDiffusionXLPipeline.from_pretrained(model_id, torch_dtype=torch.float16, use_safetensors=True,low_cpu_mem_usage=True).to("cpu")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate_image():
    prompt = request.form.get('prompt')
    negative_prompt = "blurry, low quality, close up, ugly, out of frame"
    
    image = pipe(prompt=prompt, negative_prompt=negative_prompt, num_inference_steps=30).images[0]
    image.save("static/output.png")
    
    return render_template('index.html', generated=True)

@app.route('/output')
def output_image():
    return send_file('static/output.png', mimetype='image/png')

if __name__ == "__main__":
    app.run(debug=True)
