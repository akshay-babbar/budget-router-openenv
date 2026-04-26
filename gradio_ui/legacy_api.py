from __future__ import annotations

from typing import Dict, Optional, Tuple

import requests

BASE_URL = "http://localhost:8000"
AUTO_PLAY_DELAY = 0.5


class APIClient:
    """Single-responsibility HTTP client for the OpenEnv Budget Router API."""

    def __init__(self, base_url: str = BASE_URL) -> None:
        self.base_url = base_url.rstrip("/")

    def _post(self, path: str, body: Dict) -> Tuple[Optional[Dict], Optional[str]]:
        try:
            r = requests.post(f"{self.base_url}{path}", json=body, timeout=15)
            r.raise_for_status()
            return r.json(), None
        except Exception as exc:
            return None, str(exc)

    def _get(self, path: str) -> Tuple[Optional[Dict], Optional[str]]:
        try:
            r = requests.get(f"{self.base_url}{path}", timeout=10)
            r.raise_for_status()
            return r.json(), None
        except Exception as exc:
            return None, str(exc)

    @staticmethod
    def _normalize(payload: Dict):
        """Handle both flat and observation-wrapped response shapes."""
        obs = payload.get("observation", payload)
        reward = float(payload.get("reward", obs.get("reward", 0.0)) or 0.0)
        meta = payload.get("metadata", obs.get("metadata", {})) or {}
        done = bool(payload.get("done", obs.get("done", False)))
        return obs, reward, meta, done

    def reset(self, seed: int, scenario: str):
        data, err = self._post("/reset", {"seed": seed, "scenario": scenario})
        if err:
            return None, err
        obs, _, _, _ = self._normalize(data)
        return obs, None

    def step(self, action_type: str):
        data, err = self._post("/step", {"action_type": action_type})
        if err:
            return None, err
        return self._normalize(data), None

    def state(self):
        return self._get("/state")
