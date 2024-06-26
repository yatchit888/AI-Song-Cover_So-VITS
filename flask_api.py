import io
import logging

import soundfile
import torch
import torchaudio
from flask import Flask, request, send_file
from flask_cors import CORS

from inference.infer_tool import RealTimeVC, Svc

app = Flask(__name__)

CORS(app)

logging.getLogger('numba').setLevel(logging.WARNING)


@app.route("/voiceChangeModel", methods=["POST"])
def voice_change_model():
    request_form = request.form
    wave_file = request.files.get("sample", None)
    f_pitch_change = float(request_form.get("fPitchChange", 0))
    daw_sample = int(float(request_form.get("sampleRate", 0)))
    speaker_id = int(float(request_form.get("sSpeakId", 0)))
    input_wav_path = io.BytesIO(wave_file.read())

    if raw_infer:
        # out_audio, out_sr = svc_model.infer(speaker_id, f_pitch_change, input_wav_path)
        out_audio, out_sr = svc_model.infer(speaker_id, f_pitch_change, input_wav_path, cluster_infer_ratio=0,
                                            auto_predict_f0=False, noice_scale=0.4, f0_filter=False)
        tar_audio = torchaudio.functional.resample(out_audio, svc_model.target_sample, daw_sample)
    else:
        out_audio = svc.process(svc_model, speaker_id, f_pitch_change, input_wav_path, cluster_infer_ratio=0,
                                auto_predict_f0=False, noice_scale=0.4, f0_filter=False)
        tar_audio = torchaudio.functional.resample(torch.from_numpy(out_audio), svc_model.target_sample, daw_sample)
    out_wav_path = io.BytesIO()
    soundfile.write(out_wav_path, tar_audio.cpu().numpy(), daw_sample, format="wav")
    out_wav_path.seek(0)
    return send_file(out_wav_path, download_name="temp.wav", as_attachment=True)


if __name__ == '__main__':
    raw_infer = True
    model_name = "logs/32k/G_174000-Copy1.pth"
    config_name = "configs/config.json"
    cluster_model_path = "logs/44k/kmeans_10000.pt"
    svc_model = Svc(model_name, config_name, cluster_model_path=cluster_model_path)
    svc = RealTimeVC()
    app.run(port=6842, host="0.0.0.0", debug=False, threaded=False)
