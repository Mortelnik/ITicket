"""Отсечение запросов к секретам и ограничение частоты по адресу клиента."""

from __future__ import annotations

import math
import os
import threading
import time
from collections import OrderedDict, deque
from urllib.parse import unquote


def is_secret_path(path: str) -> bool:
    # ASGI уже декодирует URL; повторная нормализация закрывает двойное
    # кодирование, Windows-разделители и разный регистр запросов сканеров.
    for _ in range(5):
        decoded = unquote(path)
        if decoded == path:
            break
        path = decoded
    parts = path.lower().replace("\\", "/").split("/")
    for part in parts:
        part = part.split(";", 1)[0].rstrip(". ")
        if part.startswith((".env", ".git", ".aws", ".ssh", ".kube")):
            return True
        if part in {".netrc", ".npmrc", ".pypirc", ".dockercfg", ".docker"}:
            return True
        if part in {"wp-config.php", "config.php", "id_rsa", "id_dsa", "id_ecdsa", "id_ed25519"}:
            return True
        if part.startswith(("wp-config.php.", "config.php.", "id_rsa.", "id_ed25519.")):
            return True
        if part.startswith(("credentials", "secrets")) or part.endswith((".pem", ".key", ".p12", ".pfx")):
            return True
    return False


class RequestGuard:
    """Скользящее окно с ограниченной памятью; состояние локально для процесса."""

    def __init__(self, limit: int = 60, window: int = 60, max_clients: int = 10_000):
        if min(limit, window, max_clients) <= 0:
            raise ValueError("Request guard settings must be positive")
        self.limit = limit
        self.window = window
        self.max_clients = max_clients
        self._clients: OrderedDict[str, deque[float]] = OrderedDict()
        self._lock = threading.Lock()

    @classmethod
    def from_env(cls) -> RequestGuard:
        return cls(
            limit=int(os.environ.get("TICKET_ROUTER_RATE_LIMIT", "60")),
            window=int(os.environ.get("TICKET_ROUTER_RATE_WINDOW", "60")),
            max_clients=int(os.environ.get("TICKET_ROUTER_RATE_MAX_CLIENTS", "10000")),
        )

    def retry_after(self, client: str) -> int:
        """0 разрешает запрос; положительное число задаёт Retry-After в секундах."""
        with self._lock:
            now = time.monotonic()
            cutoff = now - self.window
            # Порядок — по последнему разрешённому запросу. Активный клиент
            # не может вытеснить другого и обнулить его квоту.
            while self._clients and next(iter(self._clients.values()))[-1] <= cutoff:
                self._clients.popitem(last=False)
            history = self._clients.get(client)
            if history is None:
                if len(self._clients) >= self.max_clients:
                    oldest = next(iter(self._clients.values()))
                    return max(1, math.ceil(oldest[-1] + self.window - now))
                history = deque()
                self._clients[client] = history
            while history and history[0] <= cutoff:
                history.popleft()
            if len(history) >= self.limit:
                return max(1, math.ceil(history[0] + self.window - now))
            history.append(now)
            self._clients.move_to_end(client)
            return 0
