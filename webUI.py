import glob
import json
import logging
import os
import re
import subprocess
import sys
import time
import traceback
from itertools import chain
from pathlib import Path

# os.system("wget -P cvec/ https://huggingface.co/spaces/innnky/nanami/resolve/main/checkpoint_best_legacy_500.pt")
import gradio as gr
import librosa
import numpy as np
import soundfile
import torch

from compress_model import removeOptimizer
from edgetts.tts_voices import SUPPORTED_LANGUAGES
from inference.infer_tool import Svc
from utils import mix_model

logging.getLogger('numba').setLevel(logging.WARNING)
logging.getLogger('markdown_it').setLevel(logging.WARNING)
logging.getLogger('urllib3').setLevel(logging.WARNING)
logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.getLogger('multipart').setLevel(logging.WARNING)

model = None
spk = None
debug = False

local_model_root = './trained'

cuda = {}
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        device_name = torch.cuda.get_device_properties(i).name
        cuda[f"CUDA:{i} {device_name}"] = f"cuda:{i}"

def upload_mix_append_file(files,sfiles):
    try:
        if(sfiles is None):
            file_paths = [file.name for file in files]
        else:
            file_paths = [file.name for file in chain(files,sfiles)]
        p = {file:100 for file in file_paths}
        return file_paths,mix_model_output1.update(value=json.dumps(p,indent=2))
    except Exception as e:
        if debug:
            traceback.print_exc()
        raise gr.Error(e)

def mix_submit_click(js,mode):
    try:
        assert js.lstrip()!=""
        modes = {"convex combination":0, "linear combination":1}
        mode = modes[mode]
        data = json.loads(js)
        data = list(data.items())
        model_path,mix_rate = zip(*data)
        path = mix_model(model_path,mix_rate,mode)
        return f"Success, the file was saved in the{path}"
    except Exception as e:
        if debug:
            traceback.print_exc()
        raise gr.Error(e)

def updata_mix_info(files):
    try:
        if files is None :
            return mix_model_output1.update(value="")
        p = {file.name:100 for file in files}
        return mix_model_output1.update(value=json.dumps(p,indent=2))
    except Exception as e:
        if debug:
            traceback.print_exc()
        raise gr.Error(e)

def modelAnalysis(model_path,config_path,cluster_model_path,device,enhance,diff_model_path,diff_config_path,only_diffusion,use_spk_mix,local_model_enabled,local_model_selection):
    global model
    try:
        device = cuda[device] if "CUDA" in device else device
        cluster_filepath = os.path.split(cluster_model_path.name) if cluster_model_path is not None else "no_cluster"
        # get model and config path
        if (local_model_enabled):
            # local path
            model_path = glob.glob(os.path.join(local_model_selection, '*.pth'))[0]
            config_path = glob.glob(os.path.join(local_model_selection, '*.json'))[0]
        else:
            # upload from webpage
            model_path = model_path.name
            config_path = config_path.name
        fr = ".pkl" in cluster_filepath[1]
        model = Svc(model_path,
                config_path,
                device=device if device != "Auto" else None,
                cluster_model_path = cluster_model_path.name if cluster_model_path is not None else "",
                nsf_hifigan_enhance=enhance,
                diffusion_model_path = diff_model_path.name if diff_model_path is not None else "",
                diffusion_config_path = diff_config_path.name if diff_config_path is not None else "",
                shallow_diffusion = True if diff_model_path is not None else False,
                only_diffusion = only_diffusion,
                spk_mix_enable = use_spk_mix,
                feature_retrieval = fr
                )
        spks = list(model.spk2id.keys())
        device_name = torch.cuda.get_device_properties(model.dev).name if "cuda" in str(model.dev) else str(model.dev)
        msg = f"Successfully loaded model to device{device_name}上\n"
        if cluster_model_path is None:
            msg += "Clustering model or feature retrieval model not loaded\n"
        elif fr:
            msg += f"Feature Retrieval Model{cluster_filepath[1]}Loaded successfully\n"
        else:
            msg += f"clustering model{cluster_filepath[1]}Loaded successfully\n"
        if diff_model_path is None:
            msg += "Unloaded diffusion model\n"
        else:
            msg += f"diffusion model{diff_model_path.name}Loaded successfully\n"
        msg += "Currently available tones for the model：\n"
        for i in spks:
            msg += i + " "
        return sid.update(choices = spks,value=spks[0]), msg
    except Exception as e:
        if debug:
            traceback.print_exc()
        raise gr.Error(e)

    
def modelUnload():
    global model
    if model is None:
        return sid.update(choices = [],value=""),"No model to uninstall!"
    else:
        model.unload_model()
        model = None
        torch.cuda.empty_cache()
        return sid.update(choices = [],value=""),"Model unloading complete!"
    
def vc_infer(output_format, sid, audio_path, truncated_basename, vc_transform, auto_f0, cluster_ratio, slice_db, noise_scale, pad_seconds, cl_num, lg_num, lgr_num, f0_predictor, enhancer_adaptive_key, cr_threshold, k_step, use_spk_mix, second_encoding, loudness_envelope_adjustment):
    global model
    _audio = model.slice_inference(
        audio_path,
        sid,
        vc_transform,
        slice_db,
        cluster_ratio,
        auto_f0,
        noise_scale,
        pad_seconds,
        cl_num,
        lg_num,
        lgr_num,
        f0_predictor,
        enhancer_adaptive_key,
        cr_threshold,
        k_step,
        use_spk_mix,
        second_encoding,
        loudness_envelope_adjustment
    )  
    model.clear_empty()
    #Build the path to save the file and save it in the results folder
    str(int(time.time()))
    if not os.path.exists("results"):
        os.makedirs("results")
    key = "auto" if auto_f0 else f"{int(vc_transform)}key"
    cluster = "_" if cluster_ratio == 0 else f"_{cluster_ratio}_"
    isdiffusion = "sovits"
    if model.shallow_diffusion:
        isdiffusion = "sovdiff"

    if model.only_diffusion:
        isdiffusion = "diff"
    
    output_file_name = 'result_'+truncated_basename+f'_{sid}_{key}{cluster}{isdiffusion}.{output_format}'
    output_file = os.path.join("results", output_file_name)
    soundfile.write(output_file, _audio, model.target_sample, format=output_format)
    return output_file

def vc_fn(sid, input_audio, output_format, vc_transform, auto_f0,cluster_ratio, slice_db, noise_scale,pad_seconds,cl_num,lg_num,lgr_num,f0_predictor,enhancer_adaptive_key,cr_threshold,k_step,use_spk_mix,second_encoding,loudness_envelope_adjustment):
    global model
    try:
        if input_audio is None:
            return "You need to upload an audio", None
        if model is None:
            return "You need to upload an model", None
        if getattr(model, 'cluster_model', None) is None and model.feature_retrieval is False:
            if cluster_ratio != 0:
                return "You need to upload an cluster model or feature retrieval model before assigning cluster ratio!", None
        #print(input_audio)    
        audio, sampling_rate = soundfile.read(input_audio)
        #print(audio.shape,sampling_rate)
        if np.issubdtype(audio.dtype, np.integer):
            audio = (audio / np.iinfo(audio.dtype).max).astype(np.float32)
        #print(audio.dtype)
        if len(audio.shape) > 1:
            audio = librosa.to_mono(audio.transpose(1, 0))
        truncated_basename = Path(input_audio).stem[:-6]
        processed_audio = os.path.join("raw", f"{truncated_basename}.wav")
        soundfile.write(processed_audio, audio, sampling_rate, format="wav")
        output_file = vc_infer(output_format, sid, processed_audio, truncated_basename, vc_transform, auto_f0, cluster_ratio, slice_db, noise_scale, pad_seconds, cl_num, lg_num, lgr_num, f0_predictor, enhancer_adaptive_key, cr_threshold, k_step, use_spk_mix, second_encoding, loudness_envelope_adjustment)

        return "Success", output_file
    except Exception as e:
        if debug:
            traceback.print_exc()
        raise gr.Error(e)

def text_clear(text):
    return re.sub(r"[\n\,\(\) ]", "", text)

def vc_fn2(_text, _lang, _gender, _rate, _volume, sid, output_format, vc_transform, auto_f0,cluster_ratio, slice_db, noise_scale,pad_seconds,cl_num,lg_num,lgr_num,f0_predictor,enhancer_adaptive_key,cr_threshold, k_step,use_spk_mix,second_encoding,loudness_envelope_adjustment):
    global model
    try:
        if model is None:
            return "You need to upload an model", None
        if getattr(model, 'cluster_model', None) is None and model.feature_retrieval is False:
            if cluster_ratio != 0:
                return "You need to upload an cluster model or feature retrieval model before assigning cluster ratio!", None
        _rate = f"+{int(_rate*100)}%" if _rate >= 0 else f"{int(_rate*100)}%"
        _volume = f"+{int(_volume*100)}%" if _volume >= 0 else f"{int(_volume*100)}%"
        if _lang == "Auto":
            _gender = "Male" if _gender == "Male" else "Female"
            subprocess.run([sys.executable, "edgetts/tts.py", _text, _lang, _rate, _volume, _gender])
        else:
            subprocess.run([sys.executable, "edgetts/tts.py", _text, _lang, _rate, _volume])
        target_sr = 44100
        y, sr = librosa.load("tts.wav")
        resampled_y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
        soundfile.write("tts.wav", resampled_y, target_sr, subtype = "PCM_16")
        input_audio = "tts.wav"
        #audio, _ = soundfile.read(input_audio)
        output_file_path = vc_infer(output_format, sid, input_audio, "tts", vc_transform, auto_f0, cluster_ratio, slice_db, noise_scale, pad_seconds, cl_num, lg_num, lgr_num, f0_predictor, enhancer_adaptive_key, cr_threshold, k_step, use_spk_mix, second_encoding, loudness_envelope_adjustment)
        os.remove("tts.wav")
        return "Success", output_file_path
    except Exception as e:
        if debug: traceback.print_exc()  # noqa: E701
        raise gr.Error(e)

def model_compression(_model):
    if _model == "":
        return "Please select the model to be compressed first"
    else:
        model_path = os.path.split(_model.name)
        filename, extension = os.path.splitext(model_path[1])
        output_model_name = f"{filename}_compressed{extension}"
        output_path = os.path.join(os.getcwd(), output_model_name)
        removeOptimizer(_model.name, output_path)
        return f"The model has been successfully saved in the{output_path}"

def scan_local_models():
    res = []
    candidates = glob.glob(os.path.join(local_model_root, '**', '*.json'), recursive=True)
    candidates = set([os.path.dirname(c) for c in candidates])
    for candidate in candidates:
        jsons = glob.glob(os.path.join(candidate, '*.json'))
        pths = glob.glob(os.path.join(candidate, '*.pth'))
        if (len(jsons) == 1 and len(pths) == 1):
            # must contain exactly one json and one pth file
            res.append(candidate)
    return res

def local_model_refresh_fn():
    choices = scan_local_models()
    return gr.Dropdown.update(choices=choices)

with gr.Blocks(
    theme=gr.themes.Base(
        primary_hue=gr.themes.colors.green,
        font=["Source Sans Pro", "Arial", "sans-serif"],
        font_mono=['JetBrains mono', "Consolas", 'Courier New']
    ),
) as app:
    with gr.Tabs():
        with gr.TabItem("Inference"):
            with gr.Row(variant="panel"):
                with gr.Column():
                    gr.Markdown(value="""
                        <font size=2> Model Setup</font>
                        """)
                    with gr.Tabs():
                        # Invisible checkbox that tracks tab status
                        local_model_enabled = gr.Checkbox(value=False, visible=False)
                        with gr.TabItem('Upload') as local_model_tab_upload:
                            with gr.Row():
                                model_path = gr.File(label="Select a model file")
                                config_path = gr.File(label="Select a config file")
                        with gr.TabItem('Local') as local_model_tab_local:
                            gr.Markdown('Models should be placed in the {local_model_root} folder')
                            local_model_refresh_btn = gr.Button('Refresh Local Model List')
                            local_model_selection = gr.Dropdown(label='Select Model Folder', choices=[], interactive=True)
                    with gr.Row():
                        diff_model_path = gr.File(label="Select Diffusion Model File")
                        diff_config_path = gr.File(label="Select Diffusion Model Configuration File")
                    cluster_model_path = gr.File(label="Select Clustering Model or Feature Retrieval File (optional)")
                    device = gr.Dropdown(label="Inference Device, default auto-selects CPU and GPU", choices=["Auto", *cuda.keys(), "cpu"], value="Auto",visible=False)
                    enhance = gr.Checkbox(value=False,visible=False)
                    only_diffusion = gr.Checkbox(value=False,visible=False)
                with gr.Column():
                    gr.Markdown(value="""
                        <font size=3>Once all files on the left have been selected (all file modules show download), click 'Load Model' to proceed:</font>
                        """)
                    model_load_button = gr.Button(value="Load Model", variant="primary")
                    model_unload_button = gr.Button(value="Unload Model", variant="primary")
                    sid = gr.Dropdown(label="Timbre (Speaker)")
                    sid_output = gr.Textbox(label="Output Message")

            with gr.Row(visible=False):
                with gr.Column(visible=False):
                    gr.Markdown(value="""
                        <font size=2> Inference Settings</font>
                        """)
                    auto_f0 = gr.Checkbox(value=True,visible=False)
                    f0_predictor = gr.Dropdown(choices=["pm", "dio", "harvest", "crepe", "rmvpe"], value="pm",visible=False)
                    vc_transform = gr.Number(value=0,visible=False)
                    slice_db = gr.Number(value=-40,visible=False)
                    cluster_ratio = gr.Number(value=0)
                    output_format = gr.Radio(choices=["wav", "flac", "mp3"], value="wav",visible=False)
                    noise_scale = gr.Number(value=0.4,visible=False)
                    k_step = gr.Slider(value=100, minimum=1, maximum=1000,visible=False)
                with gr.Column(visible=False):
                    pad_seconds = gr.Number(value=0.5,visible=False)
                    cl_num = gr.Number(value=0,visible=False)
                    lg_num = gr.Number(value=0,visible=False)
                    lgr_num = gr.Number(value=0.75,visible=False)
                    enhancer_adaptive_key = gr.Number(value=0,visible=False)
                    cr_threshold = gr.Number(value=0.05,visible=False)
                    loudness_envelope_adjustment = gr.Number(value=0,visible=False)
                    second_encoding = gr.Checkbox(value=False,visible=False)
                    use_spk_mix = gr.Checkbox(value=False, interactive=False,visible=False)
            with gr.Tabs():
                with gr.TabItem("Audio to Audio"):
                    vc_input3 = gr.Audio(label="Select Audio", type="filepath")
                    vc_submit = gr.Button("Convert Audio", variant="primary")
                with gr.TabItem("Text to Audio"):
                    text2tts = gr.Textbox(label="Enter text to be converted here. Note, it is recommended to enable F0 prediction, otherwise it will sound strange")
                    with gr.Row():
                        tts_gender = gr.Radio(label="Speaker Gender", choices=["Male", "Female"], value="Male")
                        tts_lang = gr.Dropdown(label="Select Language, Auto detects based on input text", choices=SUPPORTED_LANGUAGES, value="Auto")
                        tts_rate = gr.Slider(label="TTS Voice Speed (relative speed value)", minimum=-1, maximum=3, value=0, step=0.1)
                        tts_volume = gr.Slider(label="TTS Voice Volume (relative value)", minimum=-1, maximum=1.5, value=0, step=0.1)
                    vc_submit2 = gr.Button("Convert Text", variant="primary")
            with gr.Row():
                with gr.Column():
                    vc_output1 = gr.Textbox(label="Output Message")
                with gr.Column():
                    vc_output2 = gr.Audio(label="Output Audio", interactive=False)

        with gr.TabItem("-",visible=False):
            gr.Markdown(value="""
                        <font size=2>Tools/Lab Features</font>
                        """,visible=False)
            with gr.Tabs(visible=False):
                with gr.TabItem("Static Voice Line Blending",visible=False):
                    mix_model_path = gr.Files(label="Select models to blend",visible=False)
                    mix_model_upload_button = gr.UploadButton("Select/Append models to blend", file_count="multiple",visible=False)
                    mix_model_output1 = gr.Textbox(
                                            label="Adjust Blending Ratios, unit/%",
                                            interactive=True,
                                            visible=False
                                         )
                    mix_mode = gr.Radio(choices=["Convex Combination", "Linear Combination"], label="Blending Mode", value="Convex Combination", interactive=True,visible=False)
                    mix_submit = gr.Button("Start Voice Blending", variant="primary",visible=False)
                    mix_model_output2 = gr.Textbox(
                                            label="Output Message",visible=False
                                         )
                    mix_model_path.change(updata_mix_info, [mix_model_path], [mix_model_output1])
                    mix_model_upload_button.upload(upload_mix_append_file, [mix_model_upload_button, mix_model_path], [mix_model_path, mix_model_output1])
                    mix_submit.click(mix_submit_click, [mix_model_output1, mix_mode], [mix_model_output2])

                with gr.TabItem("Model Compression Tool",visible=False):
                    model_to_compress = gr.File(label="Upload Model",visible=False)
                    compress_model_btn = gr.Button("Compress Model", variant="primary",visible=False)
                    compress_model_output = gr.Textbox(label="Output Information", value="",visible=False)

                    compress_model_btn.click(model_compression, [model_to_compress], [compress_model_output])

    with gr.Tabs():
        # Refresh local model list
        local_model_refresh_btn.click(local_model_refresh_fn, outputs=local_model_selection)
        # Set local enabled/disabled on tab switch
        local_model_tab_upload.select(lambda: False, outputs=local_model_enabled)
        local_model_tab_local.select(lambda: True, outputs=local_model_enabled)

        vc_submit.click(vc_fn, [sid, vc_input3, output_format, vc_transform, auto_f0, cluster_ratio, slice_db, noise_scale, pad_seconds, cl_num, lg_num, lgr_num, f0_predictor, enhancer_adaptive_key, cr_threshold, k_step, use_spk_mix, second_encoding, loudness_envelope_adjustment], [vc_output1, vc_output2])
        vc_submit2.click(vc_fn2, [text2tts, tts_lang, tts_gender, tts_rate, tts_volume, sid, output_format, vc_transform, auto_f0, cluster_ratio, slice_db, noise_scale, pad_seconds, cl_num, lg_num, lgr_num, f0_predictor, enhancer_adaptive_key, cr_threshold, k_step, use_spk_mix, second_encoding, loudness_envelope_adjustment], [vc_output1, vc_output2])

        model_load_button.click(modelAnalysis, [model_path, config_path, cluster_model_path, device, enhance, diff_model_path, diff_config_path, only_diffusion, use_spk_mix, local_model_enabled, local_model_selection], [sid, sid_output])
        model_unload_button.click(modelUnload, [], [sid, sid_output])
    os.system("start http://127.0.0.1:7860")
    app.launch()



 
