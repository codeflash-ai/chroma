from typing import Any, Dict, Optional, TypeVar
from urllib.parse import quote, urlparse, urlunparse
import logging
import orjson as json
import httpx

import chromadb.errors as errors
from chromadb.config import Settings

logger = logging.getLogger(__name__)


class BaseHTTPClient:
    _settings: Settings
    pre_flight_checks: Any = None
    keepalive_secs: int = 40

    @staticmethod
    def _validate_host(host: str) -> None:
        if "/" in host:
            parsed = urlparse(host)
            if parsed.scheme not in {"http", "https"}:
                raise ValueError(
                    "Invalid URL. " f"Unrecognized protocol - {parsed.scheme}."
                )
            if not host.startswith("http"):
                raise ValueError(
                    "Invalid URL. "
                    "Seems that you are trying to pass URL as a host but without \
                      specifying the protocol. "
                    "Please add http:// or https:// to the host."
                )

    @staticmethod
    def resolve_url(
        chroma_server_host: str,
        chroma_server_ssl_enabled: Optional[bool] = False,
        default_api_path: Optional[str] = "",
        chroma_server_http_port: Optional[int] = 8000,
    ) -> str:
        # Optimization: streamline variable usage
        BaseHTTPClient._validate_host(chroma_server_host)
        skip_port = chroma_server_host.startswith("http")
        # Only parse once
        parsed = urlparse(chroma_server_host)

        # scheme
        scheme = "https" if chroma_server_ssl_enabled else (parsed.scheme or "http")

        # netloc
        net_loc = parsed.netloc
        if not net_loc:
            net_loc = parsed.hostname or chroma_server_host

        # port
        port = ""
        if not skip_port:
            port_num = (
                parsed.port if parsed.port is not None else chroma_server_http_port
            )
            if port_num is not None:
                port = f":{port_num}"

        # path
        path = parsed.path or ""
        if not path or path == net_loc:
            path = default_api_path if default_api_path else ""
        # Avoid redundant string concatenations when default_api_path is empty or already present
        else:
            if default_api_path and not path.endswith(default_api_path):
                path = path + default_api_path

        # Avoid unnecessary double slashes and quoting
        safe_path = quote(path.replace("//", "/")) if path else ""

        full_url = urlunparse((scheme, f"{net_loc}{port}", safe_path, "", "", ""))
        return full_url

    # requests removes None values from the built query string, but httpx includes it as an empty value
    T = TypeVar("T", bound=Dict[Any, Any])

    @staticmethod
    def _clean_params(params: T) -> T:
        """Remove None values from provided dict."""
        return {k: v for k, v in params.items() if v is not None}  # type: ignore

    @staticmethod
    def _raise_chroma_error(resp: httpx.Response) -> None:
        """Raises an error if the response is not ok, using a ChromaError if possible."""
        try:
            resp.raise_for_status()
            return
        except httpx.HTTPStatusError:
            pass

        chroma_error = None
        try:
            body = json.loads(resp.text)
            if "error" in body:
                if body["error"] in errors.error_types:
                    chroma_error = errors.error_types[body["error"]](body["message"])

                    trace_id = resp.headers.get("chroma-trace-id")
                    if trace_id:
                        chroma_error.trace_id = trace_id

        except BaseException:
            pass

        if chroma_error:
            raise chroma_error

        try:
            resp.raise_for_status()
        except httpx.HTTPStatusError:
            trace_id = resp.headers.get("chroma-trace-id")
            if trace_id:
                raise Exception(f"{resp.text} (trace ID: {trace_id})")
            raise (Exception(resp.text))
