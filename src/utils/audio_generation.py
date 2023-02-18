from tortoise.api import TextToSpeech
import torch
from tortoise.utils.audio import load_voice


def get_voice(text:str, tts: TextToSpeech, voice: str, quality: str) -> torch.Tensor:
    voice_samples, conditioning_latents = load_voice(voice)
    gen = tts.tts_with_preset(text, voice_samples=voice_samples, conditioning_latents=conditioning_latents,
                              preset=quality)
    return gen.squeeze(0).cpu().numpy()
