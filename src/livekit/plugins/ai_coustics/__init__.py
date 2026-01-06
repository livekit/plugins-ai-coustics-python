# Copyright © 2025 LiveKit, Inc. All rights reserved.
# Proprietary and confidential.

from typing import Optional
from .plugin import AICousticsAudioEnhancer, AudioFilterModel, VadSettings, FRAME_USERDATA_AIC_VAD_ATTRIBUTE


def audio_enhancement(
    *,
    model: AudioFilterModel = AudioFilterModel.QUAIL_L,
    vad_settings: VadSettings = VadSettings(
        lookback_buffer_size=None,
        sensitivity=None,
    ),
):
    """
    Implements a mechanism to apply [ai-coustics models](https://ai-coustics.com/) on audio data
    represented as `AudioFrame`s. In addition, each frame will be annotated with a
    FRAME_USERDATA_AIC_VAD_ATTRIBUTE `userdata` attribute containing the output of the aic vad model.
    """
    return AICousticsAudioEnhancer(model=model, vad_settings=vad_settings)


__all__ = [
    "audio_enhancement",
    "FRAME_USERDATA_AIC_VAD_ATTRIBUTE",
    "AudioFilterModel",
    "VadSettings",
]
