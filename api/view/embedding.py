import datetime
from http import HTTPStatus

import arrow
from fastapi import APIRouter, HTTPException
from fastapi import Request
from langchain.embeddings import HuggingFaceInstructEmbeddings

from api.model.embedding import TextEmbeddingRequest, TextEmbeddingResponse
from utilities.token_helper import calc_tokens

router = APIRouter(prefix="/api/embeddings", tags=["Embeddings"])


@router.post("", summary="Get Text embeddings.", response_model=TextEmbeddingResponse)
async def get_text_embeddings(req: Request, embedding_request: TextEmbeddingRequest):
    start = arrow.utcnow()

    if not embedding_request.text and not embedding_request.texts:
        raise HTTPException(status_code=HTTPStatus.BAD_REQUEST, detail="text and texts cannot be both empty.")

    embedding_ctrl: HuggingFaceInstructEmbeddings = req.app.state.embedding
    total_token = 0
    if embedding_request.text:
        embeddings = embedding_ctrl.embed_documents([embedding_request.text])
        total_token = calc_tokens(embedding_request.text)
    else:
        embeddings = embedding_ctrl.embed_documents(embedding_request.texts)
        total_token = calc_tokens(embedding_request.texts)

    now = arrow.utcnow()
    resp = TextEmbeddingResponse(
        text=embedding_request.text,
        texts=embedding_request.texts,
        embeddings=embeddings,
        start=start.isoformat(),
        end=now.isoformat(),
        took=(now - start).total_seconds(),
        token=total_token
    )
    return resp
