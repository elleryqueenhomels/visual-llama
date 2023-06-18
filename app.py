import os
import gradio as gr
from PIL import Image

import torch
from torch.distributed import is_initialized, init_process_group
from fairscale.nn.model_parallel.initialize import initialize_model_parallel

import llama


def setup_model_parallel():
    os.environ['RANK'] = '0'
    os.environ['LOCAL_RANK'] = '0'
    os.environ['WORLD_SIZE'] = '1'
    os.environ['MP'] = '1'
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '2223'
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    world_size = int(os.environ.get("WORLD_SIZE", -1))

    if not is_initialized():
        init_process_group("nccl")
        initialize_model_parallel(world_size)
        torch.cuda.set_device(local_rank)

    # seed must be the same in all processes
    torch.manual_seed(1)
    return local_rank, world_size

setup_model_parallel()

llama_dir = ""
tokenizer, model, visual_model = llama.load_model("BIAS-7B", llama_dir)
llama_adapter = llama.LLaMA(model, tokenizer, visual_model)

def multi_modal_generate(
    img_path: str,
    prompt: str,
    max_gen_len=256,
    temperature: float = 0.1,
    top_p: float = 0.75,
):
    try:
        imgs = [Image.open(img_path).convert('RGB')]
    except:
        return ""

    prompt = llama.format_prompt(prompt)
    result = llama_adapter.generate([prompt], imgs, max_gen_len, temperature, top_p)
    print(result[0])
    return result[0]


def create_multi_modal_demo():
    with gr.Blocks() as instruct_demo:
        with gr.Row():
            with gr.Column():
                img = gr.Image(label='Input', type='filepath')
                question = gr.Textbox(lines=2, label="Prompt")
                max_len = gr.Slider(minimum=1, maximum=512,
                                    value=256, label="Max length")
                with gr.Accordion(label='Advanced options', open=False):
                    temp = gr.Slider(minimum=0, maximum=1,
                                     value=0.1, label="Temperature")
                    top_p = gr.Slider(minimum=0, maximum=1,
                                      value=0.75, label="Top p")

                run_botton = gr.Button("Run")

            with gr.Column():
                outputs = gr.Textbox(lines=10, label="Output")

        inputs = [img, question, max_len, temp, top_p]
        run_botton.click(fn=multi_modal_generate,
                         inputs=inputs, outputs=outputs)
    return instruct_demo


description = """
# LLaMA Multimodal
"""

with gr.Blocks(css="h1,p {text-align: center;}") as demo:
    gr.Markdown(description)
    with gr.TabItem("Multi-Modal Interaction"):
        create_multi_modal_demo()

demo.queue(api_open=True, concurrency_count=1).launch(share=True)
