# Copyright © 2025 LiveKit, Inc. All rights reserved.
# Proprietary and confidential.

import warnings
from importlib.metadata import version
from typing import Optional

from livekit.agents import Plugin

from .auth import Auth, AuthBase, AiCousticsApi
from .log import logger
from .plugin import AICousticsAudioEnhancer, EnhancerModel, ModelParameters, VadSettings, FRAME_USERDATA_AIC_VAD_ATTRIBUTE
from .vad import VAD


def audio_enhancement(
    *,
    model: EnhancerModel = EnhancerModel.QUAIL_L,
    vad_settings: VadSettings = VadSettings(
        speech_hold_duration=None,
        sensitivity=None,
        minimum_speech_duration=None,
    ),
    model_parameters: Optional[ModelParameters] = None,
    auth: AuthBase = Auth.livekit_cloud(),
):
    """
    Implements a mechanism to apply [ai-coustics models](https://ai-coustics.com/) on audio data
    represented as `AudioFrame`s. In addition, each frame will be annotated with a
    FRAME_USERDATA_AIC_VAD_ATTRIBUTE `userdata` attribute containing the output of the aic vad model.
    """
    if model == EnhancerModel.SPARROW_S:
        warnings.warn(
            "sparrowS is deprecated, use rookS instead",
            DeprecationWarning,
            stacklevel=2,
        )
    return AICousticsAudioEnhancer(
        model=model,
        vad_settings=vad_settings,
        model_parameters=model_parameters,
        auth=auth,
    )


class AICousticsPlugin(Plugin):
    def __init__(self) -> None:
        super().__init__(
            __name__,
            version("livekit-plugins-ai-coustics"),
            __package__ or __name__,
            logger,
        )


Plugin.register_plugin(AICousticsPlugin())

__all__ = [
    "audio_enhancement",
    "FRAME_USERDATA_AIC_VAD_ATTRIBUTE",
    "EnhancerModel",
    "VadSettings",
    "VAD",
    "ModelParameters",
    "LiveKitCloud",
    "AiCousticsApi",
    "Auth",
    "AuthBase",
]
