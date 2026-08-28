from ai_historian.model_config import resolve_chat_config


def ensure_chat_config() -> str:
    return resolve_chat_config().api_key


if __name__ == "__main__":
    config = resolve_chat_config()
    print(f"Chat configuration ready: provider={config.provider}, model={config.model}.")
