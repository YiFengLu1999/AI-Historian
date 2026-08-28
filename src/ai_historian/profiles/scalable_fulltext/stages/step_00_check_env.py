from ai_historian.model_config import (
    embedding_base_url,
    embedding_model,
    resolve_chat_config,
)


def ensure_llm_api_key() -> str:
    return resolve_chat_config().api_key


if __name__ == "__main__":
    config = resolve_chat_config()
    print(f"聊天服务商: {config.provider}")
    print(f"聊天凭证: {config.api_key_env}")
    print(f"聊天模型: {config.model}")
    print(f"聊天 base_url: {config.base_url}")
    if embedding_model() or embedding_base_url():
        print(f"Embedding 模型: {embedding_model()}")
        print(f"Embedding base_url: {embedding_base_url()}")
