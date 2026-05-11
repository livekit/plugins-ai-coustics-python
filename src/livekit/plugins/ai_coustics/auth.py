# Copyright © 2025 LiveKit, Inc. All rights reserved.
# Proprietary and confidential.

from abc import ABC, abstractmethod


class AuthBase(ABC):
    """Base for ai-coustics authentication providers."""

    provider: str

    @abstractmethod
    def _to_auth_mode(self, credentials):
        ...


class LiveKitCloud(AuthBase):
    """Use LiveKit Cloud for ai-coustics authentication and billing (default)."""

    provider = "livekitCloud"

    def _to_auth_mode(self, credentials):
        from ._ffi import AuthMode
        if credentials is None:
            return None
        return AuthMode.LIVE_KIT_CLOUD(url=credentials.url, token=credentials.token)


class AiCousticsApi(AuthBase):
    """Use your own ai-coustics credentials directly, bypassing LiveKit Cloud."""

    provider = "aiCousticsApi"

    def __init__(self, *, license_key: str):
        self.license_key = license_key

    def _to_auth_mode(self, credentials):
        from ._ffi import AuthMode
        return AuthMode.AI_COUSTICS_API(license_key=self.license_key)


class Auth:
    """Namespace of auth provider factories."""

    @staticmethod
    def livekit_cloud() -> LiveKitCloud:
        """Use LiveKit Cloud for ai-coustics authentication and billing (default)."""
        return LiveKitCloud()

    @staticmethod
    def ai_coustics_api(*, license_key: str) -> AiCousticsApi:
        """Use your own ai-coustics credentials directly, bypassing LiveKit Cloud."""
        return AiCousticsApi(license_key=license_key)
