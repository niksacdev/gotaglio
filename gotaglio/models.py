from abc import ABC, abstractmethod
from typing import Any, cast

from pydantic import BaseModel, Field

from .constants import app_configuration
from .exceptions import ExceptionContext
from .lazy_imports import azure_ai_inference, azure_core_credentials, openai
from .shared import read_data_file


class ModelSettings(BaseModel):
    """Settings for model inference requests.

    Default values work for most OpenAI models. Set to None to disable
    a parameter for models that don't support it.
    """
    max_completion_tokens: int = Field(default=800, ge=1, le=128000)
    max_tokens: int | None = Field(default=None, ge=1, le=128000)
    temperature: float | None = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float | None = Field(default=0.95, ge=0.0, le=1.0)
    frequency_penalty: float | None = Field(default=0.0, ge=-2.0, le=2.0)
    presence_penalty: float | None = Field(default=0.0, ge=-2.0, le=2.0)

    @classmethod
    def from_config(cls, config: dict) -> "ModelSettings":
        """Extract settings from model configuration.

        Uses defaults unless explicitly set in config.
        Set a value to null in JSON to disable that parameter.
        """
        # Extract only the settings fields from config
        settings_fields = {
            k: v for k, v in config.items()
            if k in cls.model_fields
        }
        return cls(**settings_fields)

    def to_api_params(self) -> dict:
        """Convert to API parameters, excluding None values."""
        params = {}

        # Handle token limit - prefer max_completion_tokens for newer models
        if self.max_completion_tokens is not None:
            params["max_completion_tokens"] = self.max_completion_tokens
        elif self.max_tokens is not None:
            params["max_tokens"] = self.max_tokens
        else:
            params["max_completion_tokens"] = 800  # sensible default

        # Add optional parameters only if set
        if self.temperature is not None:
            params["temperature"] = self.temperature
        if self.top_p is not None:
            params["top_p"] = self.top_p
        if self.frequency_penalty is not None:
            params["frequency_penalty"] = self.frequency_penalty
        if self.presence_penalty is not None:
            params["presence_penalty"] = self.presence_penalty

        return params


class Model(ABC):
    # `context` parameter provides entire test case context to
    # assist in implementing mocks that can pull the expected
    # value ouf of the context. Real models ignore the `context`
    # parameter.
    @abstractmethod
    @abstractmethod
    async def infer(self, messages, context=None) -> str:
        pass

    @abstractmethod
    def metadata(self) -> dict[str, Any]:
        pass


class AzureAI(Model):
    def __init__(self, registry, configuration):
        self._config = configuration
        self._client = None
        registry.register_model(configuration["name"], self)

    async def infer(self, messages, context=None):
        if not self._client:
            endpoint = self._config["endpoint"]
            key = self._config["key"]
            self._client = azure_ai_inference.ChatCompletionsClient(
                endpoint=endpoint,
                credential=azure_core_credentials.AzureKeyCredential(key),
            )

        response = self._client.complete(messages=messages)

        return cast(str, response.choices[0].message.content)

    def metadata(self):
        return {k: v for k, v in self._config.items() if k != "key"}


class AzureOpenAI(Model):
    def __init__(self, registry, configuration):
        self._config = configuration
        self._settings = ModelSettings.from_config(configuration)
        self._client = None
        registry.register_model(configuration["name"], self)

    async def infer(self, messages, context=None):
        if not self._client:
            self._client = openai.AzureOpenAI(
                api_key=self._config["key"],
                api_version=self._config["api"],
                azure_endpoint=self._config["endpoint"],
            )

        response = self._client.chat.completions.create(
            model=self._config["deployment"],
            messages=messages,
            **self._settings.to_api_params(),
        )

        return cast(str, response.choices[0].message.content)

    def metadata(self):
        return {k: v for k, v in self._config.items() if k != "key"}


def register_models(registry):
    config_files = app_configuration["model_config_files"]
    credentials_files = app_configuration["model_credentials_files"]

    # Read the model configuration file
    config = None
    for config_file in config_files:
        config = read_data_file(config_file, True, True)
        if config:
            break

    # Read the credentials file
    credentials = None
    for credentials_file in credentials_files:
        credentials = read_data_file(credentials_file, True, True)
        if credentials:
            break

    if config and credentials:
        # Merge in keys from credentials file
        for model in config:
            if model["name"] in credentials:
                model["key"] = credentials[model["name"]]

    if config:
        # Construct and register models.
        # TODO: lazy construction of models on first use
        for model in config:
            with ExceptionContext(f"While registering model '{model['name']}':"):
                if model["type"] == "AZURE_AI":
                    AzureAI(registry, model)
                elif model["type"] == "AZURE_OPEN_AI":
                    AzureOpenAI(registry, model)
                else:
                    raise ValueError(
                        f"Model {model['name']} has unsupported model type: {model['type']}"
                    )
