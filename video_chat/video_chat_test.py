import time
from typing import List

# def download_model():
# import urllib.request
# from pathlib import Path
# import os
# from git import Repo
# import huggingface_hub
# Path("/root/application/").mkdir(parents=True, exist_ok=True)
# os.chdir('/root/application/')
# Repo.clone_from("https://github.com/OpenGVLab/Ask-Anything.git", "/root/application/ask_anything")
# Path("/home/dhruva.patil/gale_ms/Ask-Anything/video_chat/model/").mkdir(parents=True, exist_ok=True)
# Path("/home/dhruva.patil/gale_ms/Ask-Anything/video_chat/model/stable-vicuna-7b").mkdir(parents=True, exist_ok=True)
# Path("/home/dhruva.patil/gale_ms/Ask-Anything/video_chat/model/videochat").mkdir(parents=True, exist_ok=True)
# huggingface_hub.snapshot_download(
#     'dhruva-g/video-chat-v7b',
#     local_dir='/home/dhruva.patil/gale_ms/Ask-Anything/video_chat/model/vicuna-7b',
# )
# huggingface_hub.snapshot_download(
#     'dhruva-g/video-chat-videochat_7b',
#     local_dir='/home/dhruva.patil/gale_ms/Ask-Anything/video_chat/model/videochat',
# )
# file_downloader = urllib.request
# print('Downloading files')
# file_downloader.urlretrieve(
#     "https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/eva_vit_g.pth",
#     "/home/dhruva.patil/gale_ms/Ask-Anything/video_chat/model/eva_vit_g.pth",
# )
# print('Downloaded eva_vit_g.pth')
# file_downloader.urlretrieve(
#     "https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/blip2_pretrained_flant5xxl.pth",
#     "/home/dhruva.patil/gale_ms/Ask-Anything/video_chat/model/blip2_pretrained_flant5xxl.pth",
# )
# print('Downloaded blip2_pretrained_flant5xxl.pth')
with open("/home/dhruva.patil/gale_ms/Ask-Anything/video_chat/configs/config_custom.json", "w") as file:
    file.write(
        """
{
    "model": {
        "vit_model": "eva_clip_g",
        "vit_model_path": "/home/dhruva.patil/gale_ms/Ask-Anything/video_chat/model/eva_vit_g.pth",
        "q_former_model_path": "/home/dhruva.patil/gale_ms/Ask-Anything/video_chat/model/blip2_pretrained_flant5xxl.pth",
        "llama_model_path": "/home/dhruva.patil/gale_ms/Ask-Anything/video_chat/model/vicuna-7b",
        "videochat_model_path": "/home/dhruva.patil/gale_ms/Ask-Anything/video_chat/model/videochat/videochat_7b.pth",
        "img_size": 224,
        "num_query_token": 32,
        "drop_path_rate": 0.0,
        "use_grad_checkpoint": false,
        "vit_precision": "fp32",
        "freeze_vit": true,
        "freeze_mhra": true,
        "freeze_qformer": true,
        "low_resource": false,
        "max_txt_len": 320,
        "temporal_downsample": false,
        "no_lmhra": true,
        "double_lmhra": false,
        "lmhra_reduction": 2.0,
        "gmhra_layers": 12,
        "gmhra_drop_path_rate": 0.0,
        "gmhra_dropout": 0.5,
        "extra_num_query_token": 64
    },
    "device": "cuda"
}
        """
    )
    print('Wrote config file')

import torch
from utils.config import Config
from models.videochat import VideoChat
import torch.nn as nn
from models.modeling_llama import LlamaForCausalLM
from accelerate import Accelerator
#accelerator = Accelerator(cpu=True)
#device = accelerator.device

with torch.no_grad():
    print('Initializing VideoChat')
    config_file = "configs/config_custom.json"
    cfg = Config.from_file(config_file)
    model = VideoChat(config=cfg.model)
    model = model.to('cuda')
    model = model.eval()
    print('Initialization Finished')
    print('Completed setup')
    # model.llama_model = LlamaForCausalLM.from_pretrained(
    #     cfg.model.get("llama_model_path"),
    #     torch_dtype=torch.float16,
    #     #device_map="auto",
    #     #max_memory={0: "5GIB", "cpu": "15GIB"}
    # )
    # for name, param in model.llama_model.named_parameters():
    #     param.requires_grad = False
    # model.llama_proj = nn.Linear(
    #     model.Qformer.config.hidden_size,
    #     model.llama_model.config.hidden_size
    # )
    # model = model.to('cuda')



#model = accelerator.prepare(model)

def reset_models():
    import gc;
    gc.collect()
    global model
    model.to("cpu")
    del model
    gc.collect()
    torch.cuda.empty_cache()
    config_file = "configs/config_custom.json"
    cfg = Config.from_file(config_file)
    model = VideoChat(config=cfg.model)
    model = model.to('cuda')
    model = model.eval()



def move_models(obj: VideoChat, destination: str) -> None:
    start = time.time()
    import gc;
    gc.collect()
    torch.cuda.empty_cache()
    print(f'Moving models to {destination}')
    obj.ln_vision = obj.ln_vision.to(destination)
    obj.visual_encoder = obj.visual_encoder.to(destination)
    obj.Qformer = obj.Qformer.to(destination)
    print(f'Moved models to {destination} in {time.time() - start} s')
    torch.cuda.empty_cache()





def get_answer_video(
    video_url: str,
    prompt: str,
    num_segments: int = 8,
    num_beams: int = 1,
    temperature: int = 1,
    max_new_tokens: int = 300,
) -> List[str]:
    import torch
    torch.cuda.empty_cache()
    with torch.no_grad():
        import time
        print(f'Is cuda available? {torch.cuda.is_available()}')
        print(
            f'Received request: video_url={video_url}, prompt={prompt}, num_segments={num_segments}, num_beams={num_beams}, temperature={temperature}'
        )
        import urllib.request
        video_path = 'test.mp4'
        urllib.request.urlretrieve(video_url, video_path)
        print('Downloaded video')
        img_list = []
        print('Initialization Finished')
        print('Created chat')
        from conversation import Chat
        from utils.easydict import EasyDict
        chat = Chat(model)
        chat_state = EasyDict({"system": "", "roles": ("Human", "Assistant"), "messages": [], "sep": "###"})
        start_time = time.time()
        move_models(model, 'cuda')
        chat.upload_video(video_path, chat_state, img_list, num_segments)
        move_models(model, 'cpu')
        # llm_message = chat.upload_video_without_audio(video_path, chat_state, img_list)
        print(f'Uploaded video to model {time.time() - start_time} s')
        chat_state = chat.ask(prompt, chat_state)
        try:
            # move_models(chat.model, 'cpu')
            print(f'Fetching answer for {prompt}')
            llm_message, llm_message_token, chat_state = chat.answer(
                conv=chat_state,
                img_list=img_list,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
                temperature=temperature,
            )
            print(f'Received answer from model in {time.time() - start_time} s')
            print('*' * 100)
            print(llm_message)
            print('*' * 100)
            return [llm_message]
        finally:
            del img_list
            torch.cuda.empty_cache()
            # try:
            #     # move_models(chat.model, 'cuda')
            # except:
            #     del chat.model
            #     del chat
            #     reset_models()


def get_answer_image(
    image_url: str,
    prompt: str,
    num_segments: int = 8,
    num_beams: int = 1,
    temperature: int = 1,
    max_new_tokens: int = 300,
) -> List[str]:
    import torch
    with torch.no_grad():
        import time
        print(f'Is cuda available? {torch.cuda.is_available()}')
        print(
            f'Received request: image_url={image_url}, prompt={prompt}, num_segments={num_segments}, num_beams={num_beams}, temperature={temperature}'
        )
        import urllib.request
        image_path = 'test.png'
        urllib.request.urlretrieve(image_url, image_path)
        print('Downloaded video')
        img_list = []
        print('Initialization Finished')
        print('Created chat')
        from conversation import Chat
        from utils.easydict import EasyDict
        chat = Chat(model)
        chat_state = EasyDict({"system": "", "roles": ("Human", "Assistant"), "messages": [], "sep": "###"})
        start_time = time.time()
        chat.upload_img(image_path, chat_state, img_list)
        # llm_message = chat.upload_video_without_audio(video_path, chat_state, img_list)
        print('Uploaded video to model')
        chat_state = chat.ask(prompt, chat_state)
        print(f'Fetching answer for {prompt}')
        llm_message, llm_message_token, chat_state = chat.answer(
            conv=chat_state,
            img_list=img_list,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            temperature=temperature,
        )
        print(f'Received answer from model in {time.time() - start_time} s')
        print('*' * 100)
        print(llm_message)
        print('*' * 100)
        return [llm_message]


# generate(
#     video_url='https://dhruva2.markey.ai/api/dam/v1/preview?file_id=26653f02-83ce-4146-9c71-40a3e2985c2d',
#     prompt='Analyse the video and mention which of following categories is best suited for the video based on its tone: "Entertainment", "Humour", "Educational", "Other"?',
#     num_segments=16,
#     num_beams=10,
#     temperature=1,
# )

from flask import Flask, request

app = Flask(__name__)


@app.route('/video/analyse', methods=['POST'])
def get_answer_video_api():
    print(request)
    data = request.json or {}
    video_url = data['source_url']
    prompt = data['prompt']
    num_segments = data.get('num_segments', 16)
    num_beams = data.get('num_beams', 2)
    temperature = data.get('temperature', 1)
    max_new_tokens = data.get('max_new_tokens', 300)
    responses = get_answer_video(
        video_url=video_url,
        prompt=prompt,
        num_segments=num_segments,
        num_beams=num_beams,
        temperature=temperature,
        max_new_tokens=max_new_tokens,
    )
    print(responses)
    return {
        'choices': [{
            'text': response
        } for response in responses],
        'usage': {
            'prompt_tokens': 0,
            'completion_tokens': 0,
            'total_tokens': 0,
        }
    }


@app.route('/image/analyse', methods=['POST'])
def get_answer_image_api():
    print(request)
    data = request.json or {}
    image_url = data['source_url']
    prompt = data['prompt']
    num_segments = data.get('num_segments', 16)
    num_beams = data.get('num_beams', 2)
    temperature = data.get('temperature', 1)
    max_new_tokens = data.get('max_new_tokens', 300)
    responses = get_answer_image(
        image_url=image_url,
        prompt=prompt,
        num_segments=num_segments,
        num_beams=num_beams,
        temperature=temperature,
        max_new_tokens=max_new_tokens,
    )
    print(responses)
    return {
        'choices': [{
            'text': response
        } for response in responses],
        'usage': {
            'prompt_tokens': 0,
            'completion_tokens': 0,
            'total_tokens': 0,
        }
    }


if __name__ == '__main__':
    app.run(host='0.0.0.0', port='8001')

