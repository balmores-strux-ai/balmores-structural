from pydantic import BaseModel

class Settings(BaseModel):
    app_name: str = "Structural Brain Prototype API"
    debug: bool = True
    allowed_origins: list[str] = ["http://localhost:3000"]

settings = Settings()
