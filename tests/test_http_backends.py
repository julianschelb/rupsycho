"""End-to-end tests of the API back-ends against a local fake server.

The real ``langchain-openai`` / ``langchain-ollama`` clients talk HTTP to a tiny server that
records every request. This verifies, without network access or credentials, what actually goes
over the wire: seeds, model names, concurrency and the handling of server errors.
"""

from __future__ import annotations

import copy
import json
import threading
import time
from collections.abc import Callable, Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import pytest

import rupsycho as rup


class FakeServer:
    """Records requests; answers ``ans-<seed>`` for OpenAI- and Ollama-style endpoints."""

    def __init__(self) -> None:
        self.requests: list[tuple[str, dict[str, Any]]] = []
        self.delay = 0.0
        self.fail_when: Callable[[dict[str, Any]], bool] | None = None
        self.in_flight = 0
        self.max_in_flight = 0
        self._lock = threading.Lock()
        self._httpd = ThreadingHTTPServer(("127.0.0.1", 0), self._handler())
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self._httpd.server_address[1]}"

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._httpd.shutdown()
        self._httpd.server_close()

    def _handler(self) -> type[BaseHTTPRequestHandler]:
        server = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args: Any) -> None:  # silence the default stderr logging
                pass

            def do_POST(self) -> None:  # noqa: N802 (http.server API)
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                with server._lock:
                    server.requests.append((self.path, body))
                    server.in_flight += 1
                    server.max_in_flight = max(server.max_in_flight, server.in_flight)
                try:
                    time.sleep(server.delay)
                    if server.fail_when and server.fail_when(body):
                        self._send(500, {"error": {"message": "injected failure"}})
                    elif self.path.endswith("/chat/completions"):
                        self._openai(body)
                    elif self.path.endswith("/api/generate"):
                        self._ollama(body)
                    else:
                        self._send(404, {"error": "unknown endpoint"})
                finally:
                    with server._lock:
                        server.in_flight -= 1

            def _send(self, status: int, payload: dict[str, Any], *, ndjson: bool = False) -> None:
                data = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def _openai(self, body: dict[str, Any]) -> None:
                self._send(
                    200,
                    {
                        "id": "chatcmpl-1",
                        "object": "chat.completion",
                        "created": 0,
                        "model": body["model"],
                        "choices": [
                            {
                                "index": 0,
                                "message": {
                                    "role": "assistant",
                                    "content": f"ans-{body.get('seed')}",
                                },
                                "finish_reason": "stop",
                            }
                        ],
                        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                    },
                )

            def _ollama(self, body: dict[str, Any]) -> None:
                seed = (body.get("options") or {}).get("seed")
                chunk = {"model": body["model"], "created_at": "2020-01-01T00:00:00Z"}
                lines = [
                    {**chunk, "response": f"ans-{seed}", "done": False},
                    {**chunk, "response": "", "done": True, "done_reason": "stop"},
                ]
                if body.get("stream", True):
                    self._send(200, "\n".join(json.dumps(line) for line in lines).encode())
                else:
                    self._send(200, {**lines[0], "done": True})

        return Handler


@pytest.fixture
def server() -> Iterator[FakeServer]:
    fake = FakeServer()
    fake.start()
    yield fake
    fake.stop()


def experiment_for(config_dict: dict[str, Any], models: dict[str, Any], seeds: list[str]):
    config = copy.deepcopy(config_dict)
    config["models"] = models
    config["parameters"] = {"seeds": seeds}
    return rup.experiment_from_dict(config)


def openai_model(server: FakeServer, **parameters: Any) -> dict[str, Any]:
    return {
        "gpt": {
            "type": "openai",
            "name_or_path": "gpt-test",
            "api_key": "test",
            "base_url": f"{server.url}/v1",
            "parameters": {"max_retries": 0, **parameters},
        }
    }


def ollama_model(server: FakeServer) -> dict[str, Any]:
    return {"llama": {"type": "ollama", "model": "llama-test", "base_url": server.url}}


class TestOpenAI:
    def test_seeds_and_model_name_go_over_the_wire(self, config_dict, server):
        pytest.importorskip("langchain_openai")
        experiment = experiment_for(config_dict, openai_model(server), ["11", "12"])

        summary = experiment.run(show_progress=False)

        assert summary.n_failed == 0
        assert {body["model"] for _, body in server.requests} == {"gpt-test"}
        assert {body["seed"] for _, body in server.requests} == {11, 12}
        df = experiment.get_answers_as_dataframe()
        assert (df["Answer"] == "ans-" + df["Run Seed"]).all()

    def test_calls_run_concurrently_and_keep_their_order(self, config_dict, server):
        pytest.importorskip("langchain_openai")
        server.delay = 0.1

        def run(workers):
            experiment = experiment_for(config_dict, openai_model(server), ["1"])
            start = time.perf_counter()
            experiment.run(max_concurrency=workers, show_progress=False)
            return time.perf_counter() - start, experiment.get_answers_as_dataframe()

        _, sequential_df = run(1)
        server.max_in_flight = 0
        _, parallel_df = run(8)

        assert server.max_in_flight > 1
        assert parallel_df.equals(sequential_df)

    def test_server_errors_are_isolated_per_call(self, config_dict, server):
        pytest.importorskip("langchain_openai")
        server.fail_when = lambda body: "Muller" in json.dumps(body)
        experiment = experiment_for(config_dict, openai_model(server), ["1"])

        with pytest.warns(RuntimeWarning, match="4 of 8 model calls failed"):
            summary = experiment.run(max_concurrency=4, show_progress=False)

        assert (summary.n_calls, summary.n_failed) == (8, 4)
        assert "InternalServerError" in summary.errors[0]
        assert set(experiment.get_answers_as_dataframe()["Persona ID"]) == {"Conservative Persona"}

    def test_raise_policy_surfaces_the_client_error(self, config_dict, server):
        openai = pytest.importorskip("openai")
        server.fail_when = lambda body: True
        experiment = experiment_for(config_dict, openai_model(server), ["1"])
        with pytest.raises(openai.InternalServerError):
            experiment.run(on_error="raise", show_progress=False)

    def test_cumulative_prompts_reach_the_server(self, config_dict, server):
        pytest.importorskip("langchain_openai")
        config = copy.deepcopy(config_dict)
        config["prompt_template"] = {
            "type": "chat",
            "messages": [
                {"role": "system", "content": "You are {persona_description}."},
                {"role": "user", "content": "Q: {question} Options: {answer_options} A:"},
            ],
        }
        experiment = experiment_for(config, openai_model(server), ["5"])
        experiment.run(cumulative=True, show_progress=False)

        last = [body for _, body in server.requests if "Muller" in body["messages"][0]["content"]]
        user_messages = [body["messages"][-1]["content"] for body in last]
        assert "ans-5" not in user_messages[0]
        assert "A: ans-5" in user_messages[-1]  # the earlier answers are remembered


class TestOllama:
    def test_seed_is_sent_as_an_option(self, config_dict, server):
        pytest.importorskip("langchain_ollama")
        experiment = experiment_for(config_dict, ollama_model(server), ["21", "22"])

        summary = experiment.run(show_progress=False)

        assert summary.n_failed == 0, summary.errors
        generate = [body for path, body in server.requests if path.endswith("/api/generate")]
        assert {body["options"]["seed"] for body in generate} == {21, 22}
        assert all("seed" not in body for body in generate)
        df = experiment.get_answers_as_dataframe()
        assert (df["Answer"] == "ans-" + df["Run Seed"]).all()
