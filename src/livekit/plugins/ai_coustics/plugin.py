# Copyright © 2025 LiveKit, Inc. All rights reserved.
# Proprietary and confidential.

from ._ffi import (
    Enhancer,
    EnhancerSettings,
    ModelParameters as ModelParametersUniffi,
    EnhancerModel,
    EnhancerError,
    StreamInfo,
    Credentials,
    NativeAudioBufferMut,
    VadSettings,
    model_parameters_equal,
    on_process_exit,
)
from .auth import Auth, AuthBase, LiveKitCloud
from .log import logger
from livekit import rtc
from livekit.agents import JobExecutorType, get_job_context
from typing import Optional
from dataclasses import dataclass
import asyncio
import atexit
import numpy as np

"""
The longest on_process_exit may hold up the exit of the process. Kept well inside livekit-agents'
default shutdown_process_timeout of 10s.
"""
_PROCESS_EXIT_TIMEOUT_MS = 5000

_job_shutdown_hook_registered = False


def _handle_process_exit_on_job_shutdown() -> None:
    """
    With the livekit-agents process executor, the job's process exits right after the job, and on
    Linux it exits without running atexit handlers, so the exit is handled at job shutdown instead.
    Other executors share the process between jobs, so they are left to atexit.
    """
    global _job_shutdown_hook_registered
    if _job_shutdown_hook_registered:
        return
    try:
        ctx = get_job_context()
    except RuntimeError:
        return
    if ctx.proc.executor_type != JobExecutorType.PROCESS:
        return
    _job_shutdown_hook_registered = True

    async def _on_job_shutdown() -> None:
        await asyncio.to_thread(on_process_exit, _PROCESS_EXIT_TIMEOUT_MS)

    ctx.add_shutdown_callback(_on_job_shutdown)


atexit.register(on_process_exit, _PROCESS_EXIT_TIMEOUT_MS)

@dataclass
class ModelParameters:
    enhancement_level: Optional[float] = None
    bypass: Optional[float] = None

    def _to_uniffi(self):
        return ModelParametersUniffi(
            enhancement_level=self.enhancement_level,
            bypass=self.bypass,
        )


def to_native_buffer(data: memoryview) -> tuple[np.ndarray, NativeAudioBufferMut]:
    """
    Convert frame.data (int16 memoryview) to NativeAudioBufferMut (f32 pointer).
    Returns both the numpy array (to keep it alive) and the NativeAudioBufferMut.
    """
    # Convert int16 to float32 in range [-1.0, 1.0]
    # astype() creates a copy, which is writable by default
    samples = (
        np.frombuffer(data, dtype=np.int16).astype(np.float32, copy=True) / 32768.0
    )

    # Get the memory address directly from the numpy array
    ptr_value = samples.ctypes.data

    # Create NativeAudioBufferMut pointing to the numpy memory
    native_buffer = NativeAudioBufferMut(
        ptr=ptr_value,
        len=len(samples),  # Number of f32 samples
    )

    return samples, native_buffer

"""
Attribute used to store associated VAD data (the return value of
https://docs.rs/aic-sdk/latest/aic_sdk/struct.EnergyVadContext.html#method.is_speech_detected) from aic
model into processed `AudioFrame`s.
"""
FRAME_USERDATA_AIC_VAD_ATTRIBUTE = "lk.aic-vad"

class AICousticsAudioEnhancer(rtc.FrameProcessor[rtc.AudioFrame]):

    def __init__(
        self,
        *,
        model: EnhancerModel,
        vad_settings: VadSettings,
        model_parameters: Optional[ModelParameters] = None,
        auth: Optional[AuthBase] = None,
    ) -> None:
        self._model = model
        self._vad_settings = vad_settings
        self._model_parameters = model_parameters
        self._auth = auth or Auth.livekit_cloud()
        self._last_error_msg: Optional[str] = None

        self._enhancer: Enhancer | None = None
        self._info: StreamInfo | None = None
        self._credentials: Credentials | None = None
        self._settings: EnhancerSettings | None = None
        self._enabled = True

        _handle_process_exit_on_job_shutdown()

    @property
    def enabled(self) -> bool:
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool) -> None:
        self._enabled = value

    def update_model_parameters(self, model_parameters: ModelParameters):
        """
        Updates the model parameters on the running model.

        The native core must already exist (i.e. at least one audio frame must
        have been processed) for the update to take effect; otherwise the call
        is a no-op and a warning is logged. The new parameters are also stored
        so they are reapplied if the native core is later recreated (e.g. on a
        sample-rate or channel change).
        """
        if not self._enhancer:
            logger.warning("update_model_parameters: Native core not yet initialized, skipping. Process at least one audio frame first.")
            return
        new_uniffi = model_parameters._to_uniffi()
        current_uniffi = (
            self._model_parameters._to_uniffi()
            if self._model_parameters is not None
            else ModelParametersUniffi(bypass=None, enhancement_level=None)
        )
        if model_parameters_equal(new_uniffi, current_uniffi):
            return
        self._model_parameters = model_parameters
        self._enhancer.update_model_parameters(new_uniffi)

    def _on_stream_info_updated(
        self, *, room_name: str, participant_identity: str, publication_sid: str
    ):
        self._info = StreamInfo(
            room_id="",
            room_name=room_name,
            participant_identity=participant_identity,
            participant_id="",
            track_id=publication_sid,
        )
        if self._enhancer is not None:
            self._enhancer.update_stream_info(self._info)

    def _on_credentials_updated(self, *, token: str, url: str):
        self._credentials = Credentials(token=token, url=url)
        if self._enhancer is not None:
            self._enhancer.update_credentials(self._credentials)

    def _process(self, frame: rtc.AudioFrame) -> rtc.AudioFrame:
        """
        Processes a single audio frame.

        If the frame processor is disabled or processing fails, the original frame is
        returned unchanged.
        """
        if not self.enabled:
            return frame

        auth_mode = self._auth._to_auth_mode(self._credentials)
        if auth_mode is None:
            self._log_process_frame_error("Missing auth mode")
            return frame

        if self._auth_mode_requires_credentials() and not self._credentials:
            self._log_process_frame_error("Missing credentials")
            return frame

        if self._auth_mode_requires_stream_info() and self._info is None:
            self._log_process_frame_error("Missing stream info")
            return frame

        ## lazily create enhancer
        if self._enhancer is None or (
            ## implicitly recreate audio enhancer on sample rate or channel changes
            self._settings is not None
            and (
                self._settings.sample_rate != frame.sample_rate
                or self._settings.num_channels != frame.num_channels
                or self._settings.samples_per_channel != frame.samples_per_channel
            )
        ):
            self._settings = EnhancerSettings(
                sample_rate=frame.sample_rate,
                num_channels=frame.num_channels,
                samples_per_channel=frame.samples_per_channel,
                model=self._model,
                model_parameters=self._model_parameters._to_uniffi() if self._model_parameters else ModelParametersUniffi(bypass=None, enhancement_level=None),
                vad=self._vad_settings
            )
            try:
                self._enhancer = Enhancer(auth_mode, self._settings)
            except EnhancerError as e:
                self._log_process_frame_error(f"Failed to initialize plugin core: {e} - Disabling noise cancellation for all following audio frames.")
                self._enhancer = None
                self._enabled = False
                return frame
            if self._info is not None:
                self._enhancer.update_stream_info(self._info)
            _handle_process_exit_on_job_shutdown()

        # Convert frame.data to NativeAudioBufferMut (f32)
        # Keep samples alive during the process call
        samples, native_buffer = to_native_buffer(frame.data)

        # Process in-place (modifies samples array)
        try:
            vad_data = self._enhancer.process_with_vad(native_buffer)
        except EnhancerError as e:
            self._log_process_frame_error(f"Processing failed: {e}")
            return frame

        # Convert back to int16 and create new frame
        processed_int16 = (np.clip(samples, -1.0, 1.0) * 32767.0).astype(np.int16)

        output_frame = rtc.AudioFrame(
            data=processed_int16.tobytes(),
            sample_rate=frame.sample_rate,
            num_channels=frame.num_channels,
            samples_per_channel=frame.samples_per_channel,
            userdata=frame.userdata,
        )
        output_frame.userdata[FRAME_USERDATA_AIC_VAD_ATTRIBUTE] = vad_data
        return output_frame

    def _auth_mode_requires_stream_info(self) -> bool:
        """Does the given auth mode require update_stream_info be called?"""
        return isinstance(self._auth, LiveKitCloud)

    def _auth_mode_requires_credentials(self) -> bool:
        """
        Does the given auth mode require update_credentials be called?

        Note that this is just here to provide helpful warnings to users,
        the actual auth layer is in the rust core.
        """
        return isinstance(self._auth, LiveKitCloud)

    def _log_process_frame_error(self, msg: str):
        """
        Logs a new error to the screen when processing a frame.
        Only shows logs which were newly introduced as compared with the
        last processed frame.
        """
        if self._last_error_msg == msg:
            return
        self._last_error_msg = msg
        logger.error(msg)

    def _close(self):
        if self._enhancer is not None:
            self._enhancer = None
