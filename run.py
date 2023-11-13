import uvicorn

from config.settings import Settings

if __name__ == '__main__':
    settings = Settings()
    uvicorn.run(
        app="api:init_app",
        host=settings.host,
        port=settings.port,
        use_colors=True,
        log_level=settings.log_level,
        timeout_keep_alive=settings.timeout_keep_alive,
        factory=True,
    )
