# VoxCPM2 Turkish TTS LoRA & Voice Clone

A Turkish LoRA adapter trained on top of [[VoxCPM2](https://huggingface.co/openbmb/VoxCPM2?utm_source=chatgpt.com)](https://huggingface.co/openbmb/VoxCPM2), using ~18 hours of single-speaker data.

## Training Details

* **Base model:** [openbmb/VoxCPM2](https://huggingface.co/openbmb/VoxCPM2)
* **Method:** LoRA adaptation 
* **LoRA rank:** 64, alpha: 64
* **Checkpoint:** step 1400 (selected based on inference experiments)
* **Preprocessing:** Turkish-specific number/date normalization (`trnorm`), -23 LUFS loudness normalization, sentence integrity filtering (truncated segments removed), 3–20 second utterance range

## Sample Inference Results
<audio controls> <source src="https://huggingface.co/mramazan/voxcpm2-turkish-tts-lora/resolve/main/1.wav" type="audio/wav"> </audio>
<audio controls> <source src="https://huggingface.co/mramazan/voxcpm2-turkish-tts-lora/resolve/main/2.wav" type="audio/wav"> </audio>
<audio controls> <source src="https://huggingface.co/mramazan/voxcpm2-turkish-tts-lora/resolve/main/3.wav" type="audio/wav"> </audio>





## Usage

```python
from voxcpm import VoxCPM
from voxcpm.core import LoRAConfig
from huggingface_hub import snapshot_download

base_model_path = snapshot_download("openbmb/VoxCPM2")
lora_path = snapshot_download("mramazan/voxcpm2-turkish-tts-lora")

lora_cfg = LoRAConfig(
    enable_lm=True,
    enable_dit=True,
    enable_proj=False,
    r=64,
    alpha=64,
    dropout=0.0,
)

model = VoxCPM.from_pretrained(
    base_model_path,
    lora_weights_path=lora_path,
    lora_config=lora_cfg,
    device="auto",
)

wav = model.generate(
    text="Your Turkish text goes here.",
    #reference_wav_path="reference_audio.wav",
    cfg_value=2.0,
    inference_timesteps=16,
    normalize=False,
)
```

## Known Limitations

* Stability issues may occur with very short (<3 seconds) or highly expressive input texts.

## License

Same license with the base model, VoxCPM2. (Apache-2.0).
