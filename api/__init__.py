import logging
import os

import torch
from fastapi import FastAPI
from langchain.embeddings import HuggingFaceInstructEmbeddings
from starlette.middleware.gzip import GZipMiddleware
from torch import hub
from transformers import AutoTokenizer, AutoModel, AutoModelForPreTraining
from config.settings import Settings
from utilities.log_util import initialize_logging
from utilities.torch_helper import torch_gc


def init_app():
    settings = Settings()
    initialize_logging(log_level=settings.log_level)
    logger = logging.getLogger(__name__)

    app = FastAPI(title=settings.api_name, version=settings.api_version, root_path=settings.openapi_prefix)
    app.add_middleware(GZipMiddleware, minimum_size=1024 * 1024)

    app.state.settings = settings
    app.state.logger = logger

    # Setup huggingface cache home dir
    # https://huggingface.co/transformers/v4.0.1/installation.html
    # https://huggingface.co/docs/huggingface_hub/guides/manage-cache
    os.environ['HUGGINGFACE_HUB_CACHE'] = settings.huggingface_home_dir
    hub.set_dir(settings.torch_hub_dir)

    # Load torch hub model from ./torch
    # For embeddings
    os.environ['SENTENCE_TRANSFORMERS_HOME'] = settings.sentence_transformers_dir

    if torch.cuda.is_available():
        logger.info("GPU Found. Running in GPU mode.")
        use_gpu = True
    else:
        logger.info("GPU Not Found. Running in CPU mode.")
        use_gpu = False

    tokenizer = AutoTokenizer.from_pretrained(settings.chatglm_model_path,
                                              trust_remote_code=True,
                                              revision="v1.0")

    if use_gpu:
        model = AutoModel.from_pretrained(settings.chatglm_model_path,
                                          trust_remote_code=True,
                                          revision="v1.0").cuda()
    else:
        model = AutoModel.from_pretrained(settings.chatglm_model_path,
                                          trust_remote_code=True,
                                          revision="v1.0").half().cpu()
    model.eval()

    app.state.model = model
    app.state.tokenizer = tokenizer
    app.state.embedding = HuggingFaceInstructEmbeddings()

    @app.on_event("startup")
    async def startup():
        pass

    @app.on_event("shutdown")
    async def shutdown():
        torch_gc()

    from api.view.default import router as default_router
    app.include_router(default_router)

    from api.view.llm import router as llm_router
    app.include_router(llm_router, prefix=settings.prefix)

    from api.view.embedding import router as emb_router
    app.include_router(emb_router, prefix=settings.prefix)

    return app
