"""Flask JSON-RPC bridge for the Mastra plagiarism agent."""

from __future__ import annotations

import logging
import os
import uuid
from typing import Any, Dict, Optional

import requests
from flask import Flask
from flask_jsonrpc import JSONRPC

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional dependency
    load_dotenv = None


logger = logging.getLogger(__name__)


class MastraAgentError(RuntimeError):
    """Raised when the Mastra agent responds with an error."""


def _build_default_payload(
    text: str,
    reference_text: Optional[str],
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build the message payload expected by Mastra's A2A endpoint."""

    parts = [{"kind": "text", "text": text}]
    if reference_text:
        parts.append(
            {
                "kind": "text",
                "text": f"Reference text provided by user:\n{reference_text}",
            }
        )

    payload: Dict[str, Any] = {
        "method": "message/send",
        "params": {
            "message": {
                "role": "user",
                "kind": "message",
                "messageId": uuid.uuid4().hex,
                "parts": parts,
            }
        },
    }

    if metadata:
        payload["params"]["metadata"] = metadata

    return payload


def call_plagiarism_agent(
    text: str,
    reference_text: Optional[str] = None,
    threshold: Optional[float] = None,
) -> Dict[str, Any]:
    """Send the user's text to the Mastra plagiarism agent via HTTP.

    This implementation assumes you have a running Mastra instance (for example,
    started with ``mastra start``) that exposes an A2A endpoint for the
    plagiarism agent. Configure the endpoint via the ``MASTRA_AGENT_URL``
    environment variable if necessary.
    """

    if not text or not text.strip():
        raise ValueError("Text to analyze must be a non-empty string")

    if load_dotenv:
        load_dotenv()

    agent_url = os.getenv(
        "MASTRA_AGENT_URL",
        "http://127.0.0.1:8787/a2a/plagiarismAgent",
    )

    # Optional metadata forwarded to the agent
    metadata: Dict[str, Any] = {}
    if threshold is not None:
        metadata["threshold"] = threshold

    payload = _build_default_payload(
        text.strip(),
        reference_text,
        metadata if metadata else None,
    )

    try:
        response = requests.post(
            agent_url,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=(5, int(os.getenv("MASTRA_AGENT_TIMEOUT", "60"))),
        )
    except requests.RequestException as exc:  # pragma: no cover - network errors
        logger.exception("Failed to reach Mastra agent")
        raise MastraAgentError("Failed to reach Mastra agent") from exc

    if response.status_code >= 400:
        logger.error(
            "Mastra agent responded with status %s: %s",
            response.status_code,
            response.text,
        )
        raise MastraAgentError(
            f"Mastra agent error (status {response.status_code}): {response.text}"
        )

    try:
        return response.json()
    except ValueError as exc:  # pragma: no cover - unexpected payload
        logger.exception("Mastra agent returned invalid JSON")
        raise MastraAgentError("Mastra agent returned invalid JSON") from exc

app = Flask(__name__)
jsonrpc = JSONRPC(app, '/api')

@jsonrpc.method('plagiarism.check')
def check_plagiarism(
    text: str,
    reference_text: Optional[str] = None,
    threshold: Optional[float] = None,
) -> dict:
    """
    This function is called via JSON-RPC 2.0.
    'text' is the user's input from Telex.im.
    """
    try:
        agent_response = call_plagiarism_agent(
            text=text,
            reference_text=reference_text,
            threshold=threshold,
        )
        return {
            "result": "success",
            "data": agent_response,
            "message": "Plagiarism check completed.",
        }
    except (ValueError, MastraAgentError) as exc:
        return {
            "result": "error",
            "data": None,
            "message": str(exc),
        }
    except Exception as exc:  # pragma: no cover - unexpected errors
        logger.exception("Unexpected error while checking plagiarism")
        return {
            "result": "error",
            "data": None,
            "message": f"An unexpected error occurred: {exc}",
        }

if __name__ == '__main__':
    port = int(os.getenv('FLASK_PORT', '3000'))
    app.run(debug=True, host='0.0.0.0', port=port)