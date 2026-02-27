# Copyright © 2025 LiveKit, Inc. All rights reserved.
# Proprietary and confidential.

from __future__ import annotations

import weakref

from livekit import agents, rtc

from .log import logger
from .plugin import FRAME_USERDATA_AIC_VAD_ATTRIBUTE

SPEECH_BUFFER_MAX_FRAMES = 3000


class VAD(agents.vad.VAD):
    """
    A VAD implementation that relies on the accompanying ai-coustics
    :func:`~livekit.plugins.ai_coustics.audio_enhancement` FrameProcessor
    instead of performing its own inference.
    """

    def __init__(self) -> None:
        super().__init__(capabilities=agents.vad.VADCapabilities(update_interval=0.032))
        self._streams = weakref.WeakSet[VADStream]()

    @property
    def model(self) -> str:
        return "ai-coustics"

    @property
    def provider(self) -> str:
        return "ai-coustics"

    def stream(self) -> VADStream:
        stream = VADStream(self)
        self._streams.add(stream)
        return stream


class VADStream(agents.vad.VADStream):
    def __init__(self, vad: VAD) -> None:
        super().__init__(vad)
        self._has_no_metadata_counter = 0

    @agents.utils.log_exceptions(logger=logger)
    async def _main_task(self) -> None:
        sample_rate = 0
        speaking = False
        speech_frame_count = 0
        silence_frame_count = 0
        speech_buffer: list[rtc.AudioFrame] = []
        current_sample = 0
        timestamp = 0.0
        speech_duration = 0.0
        silence_duration = 0.0

        async for input_frame in self._input_ch:
            if not isinstance(input_frame, rtc.AudioFrame):
                continue  # skip flush sentinel

            if not sample_rate:
                sample_rate = input_frame.sample_rate
            elif input_frame.sample_rate != sample_rate:
                logger.error("a frame with another sample rate was already pushed")
                continue

            vad_metadata = input_frame.userdata.get(FRAME_USERDATA_AIC_VAD_ATTRIBUTE)

            if vad_metadata is None:
                if len(input_frame.data) > 0:
                    # only log error after 10 consecutive frames without metadata to account for tearing down the pipeline and flushing
                    if self._has_no_metadata_counter > 10:
                        logger.error(
                            "No VAD metadata found in frame.userdata['%s'] "
                            "make sure that you are using noise_cancellation=audio_enhancement() on the audio input. "
                            "This VAD plugin relies on its preprocessing.",
                            FRAME_USERDATA_AIC_VAD_ATTRIBUTE,
                        )
                    self._has_no_metadata_counter += 1
                continue
            else:
                self._has_no_metadata_counter = 0

            is_speaking: bool = vad_metadata
            frame_duration = input_frame.duration

            current_sample += input_frame.samples_per_channel
            timestamp += frame_duration

            if speaking:
                speech_duration += frame_duration
            else:
                silence_duration += frame_duration

            # always emit INFERENCE_DONE for metrics/monitoring
            self._event_ch.send_nowait(
                agents.vad.VADEvent(
                    type=agents.vad.VADEventType.INFERENCE_DONE,
                    samples_index=current_sample,
                    timestamp=timestamp,
                    speech_duration=speech_duration,
                    silence_duration=silence_duration,
                    probability=1.0,
                    inference_duration=0.0,
                    frames=[input_frame],
                    speaking=speaking,
                    raw_accumulated_silence=silence_frame_count,
                    raw_accumulated_speech=speech_frame_count,
                )
            )

            speech_buffer.append(input_frame)

            if is_speaking:
                speech_frame_count += 1
                silence_frame_count = 0

                if not speaking:
                    speaking = True
                    silence_duration = 0.0

                    self._event_ch.send_nowait(
                        agents.vad.VADEvent(
                            type=agents.vad.VADEventType.START_OF_SPEECH,
                            samples_index=current_sample,
                            timestamp=timestamp,
                            speech_duration=speech_frame_count
                            * self._vad.capabilities.update_interval,
                            silence_duration=0.0,
                            probability=1.0,
                            inference_duration=0.0,
                            frames=list(speech_buffer),
                            speaking=True,
                            raw_accumulated_silence=0,
                            raw_accumulated_speech=speech_frame_count,
                        )
                    )
            else:
                silence_frame_count += 1
                speech_frame_count = 0

                if speaking:
                    speaking = False
                    speech_duration = 0.0

                    self._event_ch.send_nowait(
                        agents.vad.VADEvent(
                            type=agents.vad.VADEventType.END_OF_SPEECH,
                            samples_index=current_sample,
                            timestamp=timestamp,
                            speech_duration=0.0,
                            silence_duration=silence_frame_count
                            * self._vad.capabilities.update_interval,
                            probability=1.0,
                            inference_duration=0.0,
                            frames=list(speech_buffer),
                            speaking=False,
                            raw_accumulated_silence=silence_frame_count,
                            raw_accumulated_speech=0,
                        )
                    )

                    speech_buffer = []

            # keep buffer size manageable
            if len(speech_buffer) > SPEECH_BUFFER_MAX_FRAMES:
                speech_buffer = speech_buffer[-SPEECH_BUFFER_MAX_FRAMES:]
