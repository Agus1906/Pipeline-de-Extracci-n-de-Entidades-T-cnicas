"""Pruebas del contrato y LCEL; transporte de OpenAI simulado, sin claves reales."""
import json

import httpx
import pytest
from langchain_core.messages import AIMessage
from langchain_openai import ChatOpenAI
from pydantic import ValidationError

import chain
from schemas import EntidadesTecnicas

VALID = dict(tecnologias=["FastAPI", "Redis", "PostgreSQL"],
             nivel_de_criticidad="alta", resumen_tecnico="Servicio caído por agotamiento de conexiones.")


@pytest.mark.parametrize("changes", [
    {"tecnologias": []}, {"tecnologias": [" "]},
    {"tecnologias": ["Redis", "redis"]}, {"nivel_de_criticidad": "urgente"},
    {"resumen_tecnico": "  "}, {"extra": "rechazar"},
])
def test_contract_rejects_invalid_data(changes):
    with pytest.raises(ValidationError):
        EntidadesTecnicas.model_validate(VALID | changes)


@pytest.mark.parametrize("reason", ["length", "max_tokens", "content_filter", None])
def test_rejects_unfinished_even_when_parsed(reason):
    with pytest.raises(chain.InvalidModelOutput):
        chain.validate_output(dict(raw=AIMessage(content="", response_metadata={"finish_reason": reason}),
                                   parsed=VALID, parsing_error=None))


@pytest.mark.parametrize("broken", ["{invalid", "{}", json.dumps(VALID | {"tecnologias": []})])
async def test_real_parser_retries_and_recovers(broken, caplog):
    calls = []

    def respond(request):
        calls.append(json.loads(request.content))
        arguments = broken if len(calls) == 1 else json.dumps(VALID)
        return httpx.Response(200, json={
            "id": "chatcmpl-test", "object": "chat.completion", "created": 1, "model": "test",
            "choices": [{"index": 0, "finish_reason": "tool_calls", "message": {
                "role": "assistant", "content": None, "tool_calls": [{
                    "id": "call_test", "type": "function", "function": {
                        "name": "EntidadesTecnicas", "arguments": arguments}}]}}]})

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        model = ChatOpenAI(model="test", api_key="fake-test-key", http_async_client=client,
                           max_retries=0, use_responses_api=False)
        with caplog.at_level("INFO"):
            result = await chain.build_chain(model, wait=False).ainvoke(
                {"text": "FastAPI y Redis: servicio caído."},
                config={"metadata": {"request_id": "test"}})
    assert result == EntidadesTecnicas(**VALID)
    assert len(calls) == 2
    assert "Validación fallida" in caplog.text
    assert "fake-test-key" not in caplog.text
    assert calls[0]["messages"][1]["content"].endswith("FastAPI y Redis: servicio caído.")


class StubModel:
    def __init__(self, *, truncated=False, error=None):
        self.calls = 0
        self.truncated = truncated
        self.error = error

    def with_structured_output(self, schema, **kwargs):
        assert schema is EntidadesTecnicas
        assert kwargs["include_raw"] is True

        async def invoke(_):
            self.calls += 1
            if self.error:
                raise self.error
            return dict(raw=AIMessage(content="", response_metadata={
                "finish_reason": "length" if self.truncated else "stop"}),
                parsed=None, parsing_error=ValueError("invalid"))
        return chain.RunnableLambda(invoke)


@pytest.mark.parametrize("truncated", [False, True])
async def test_exhausted_retries(truncated):
    model = StubModel(truncated=truncated)
    with pytest.raises(chain.InvalidModelOutput):
        await chain.build_chain(model, wait=False).ainvoke(
            {"text": "Texto ambiguo"}, config={"metadata": {"request_id": "test"}})
    assert model.calls == 3


async def test_unrelated_errors_not_retried():
    model = StubModel(error=RuntimeError("configuration"))
    with pytest.raises(RuntimeError):
        await chain.build_chain(model, wait=False).ainvoke(
            {"text": "Redis"}, config={"metadata": {"request_id": "test"}})
    assert model.calls == 1


@pytest.mark.parametrize("text", ["", "  ", None, "x" * 20001])
async def test_invalid_input_never_builds_chain(text, monkeypatch):
    def forbidden():
        pytest.fail("No debe crear el cliente con entrada inválida")
    monkeypatch.setattr(chain, "build_chain", forbidden)
    with pytest.raises(ValueError):
        await chain.process_text(text)


async def test_process_text_uses_async_chain(monkeypatch):
    async def invoke(inputs):
        assert inputs == {"text": "Redis"}
        return EntidadesTecnicas(**VALID)
    monkeypatch.setattr(chain, "build_chain", lambda: chain.RunnableLambda(invoke))
    assert await chain.process_text(" Redis ") == EntidadesTecnicas(**VALID)
