import json
from logging import Logger

import arrow
from fastapi import APIRouter, HTTPException
from fastapi import Request
from sse_starlette import EventSourceResponse
from transformers import PreTrainedModel, PreTrainedTokenizer

from api.model.llm import ModelList, ModelCard, ChatCompletionResponse, ChatCompletionRequest, UsageInfo, ChatMessage, \
    ChatCompletionResponseChoice, ChatCompletionResponseStreamChoice, DeltaMessage
from utilities.token_helper import calc_tokens
from utils import generate_chatglm3, process_response, generate_stream_chatglm3

router = APIRouter(prefix="/api", tags=["API"])


@router.get("/v1/models", response_model=ModelList)
async def list_models():
    model_card = ModelCard(id="gpt-3.5-turbo")
    return ModelList(data=[model_card])


@router.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def create_chat_completion(req: Request, request: ChatCompletionRequest):
    # global model, tokenizer

    if request.messages[-1].role == "assistant":
        raise HTTPException(status_code=400, detail="Invalid request")

    with_function_call = bool(request.messages[0].role == "system" and request.messages[0].tools is not None)

    # stop settings
    request.stop = request.stop or []
    if isinstance(request.stop, str):
        request.stop = [request.stop]

    request.stop_token_ids = request.stop_token_ids or []

    gen_params = dict(
        messages=request.messages,
        temperature=request.temperature,
        top_p=request.top_p,
        max_tokens=request.max_tokens or 1024,
        echo=False,
        stream=request.stream,
        stop_token_ids=request.stop_token_ids,
        stop=request.stop,
        repetition_penalty=request.repetition_penalty,
        with_function_call=with_function_call,
    )

    if request.stream:
        generate = predict(request.model, req.app.state.model, req.app.state.tokenizer, gen_params)
        return EventSourceResponse(generate, media_type="text/event-stream")

    response = generate_chatglm3(req.app.state.model, req.app.state.tokenizer, gen_params)
    usage = UsageInfo()

    finish_reason, history = "stop", None
    if with_function_call and request.return_function_call:
        history = [m.dict(exclude_none=True) for m in request.messages]
        content, history = process_response(response["text"], history)
        if isinstance(content, dict):
            message, finish_reason = ChatMessage(
                role="assistant",
                content=json.dumps(content, ensure_ascii=False),
            ), "function_call"
        else:
            message = ChatMessage(role="assistant", content=content)
    else:
        message = ChatMessage(role="assistant", content=response["text"])

    choice_data = ChatCompletionResponseChoice(
        index=0,
        message=message,
        finish_reason=finish_reason,
        history=history
    )

    task_usage = UsageInfo.parse_obj(response["usage"])
    for usage_key, usage_value in task_usage.dict().items():
        setattr(usage, usage_key, getattr(usage, usage_key) + usage_value)

    return ChatCompletionResponse(model=request.model, choices=[choice_data], object="chat.completion", usage=usage)


async def predict(model_id: str, model: PreTrainedModel, tokenizer: PreTrainedTokenizer, params: dict):
    choice_data = ChatCompletionResponseStreamChoice(
        index=0,
        delta=DeltaMessage(role="assistant"),
        finish_reason=None
    )
    chunk = ChatCompletionResponse(model=model_id, choices=[choice_data], object="chat.completion.chunk")
    yield "{}".format(chunk.json(exclude_unset=True, ensure_ascii=False))

    previous_text = ""
    for new_response in generate_stream_chatglm3(model, tokenizer, params):
        decoded_unicode = new_response["text"]
        delta_text = decoded_unicode[len(previous_text):]
        previous_text = decoded_unicode

        if len(delta_text) == 0:
            delta_text = None

        choice_data = ChatCompletionResponseStreamChoice(
            index=0,
            delta=DeltaMessage(content=delta_text),
            finish_reason=None
        )
        chunk = ChatCompletionResponse(model=model_id, choices=[choice_data], object="chat.completion.chunk")
        yield "{}".format(chunk.json(exclude_unset=True, ensure_ascii=False))

    choice_data = ChatCompletionResponseStreamChoice(
        index=0,
        delta=DeltaMessage(),
        finish_reason="stop"
    )
    chunk = ChatCompletionResponse(model=model_id, choices=[choice_data], object="chat.completion.chunk")
    yield "{}".format(chunk.json(exclude_unset=True, ensure_ascii=False))
    yield '[DONE]'

