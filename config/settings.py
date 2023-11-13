from pydantic import BaseSettings


class Settings(BaseSettings):
    api_name: str = "ChatGLM3-6B API"
    api_version: str = "0.0.1"
    host: str = "localhost"  # "0.0.0.0"
    port: int = 8083
    debug: bool = True
    prefix: str = ""
    openapi_prefix: str = ""
    timeout_keep_alive: int = 120
    log_level: str = "debug"

    chatglm_model_path: str = "D:\\ai\\nlp\\llm\\THUDM\\chatglm3-6b-32k"

    torch_hub_dir: str = "D:/ai/cache/torch"
    sentence_transformers_dir: str = "D:/ai/cache/torch/sentence_transformers"
    huggingface_home_dir: str = "D:/ai/cache/huggingface"



