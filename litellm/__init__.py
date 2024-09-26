### Hide pydantic namespace conflict warnings globally ###
import warnings

warnings.filterwarnings("ignore", message=".*conflict with protected namespace.*")
### INIT VARIABLES ###
import threading, requests, os
from typing import Callable, List, Optional, Dict, Union, Any, Literal, get_args
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler
from litellm.caching import Cache
from litellm._logging import (
    set_verbose,
    _turn_on_debug,
    verbose_logger,
    json_logs,
    _turn_on_json,
    log_level,
)

from litellm.types.guardrails import GuardrailItem
from litellm.proxy._types import (
    KeyManagementSystem,
    KeyManagementSettings,
    LiteLLM_UpperboundKeyGenerateParams,
)
import httpx
import dotenv
import json
from enum import Enum
import json

litellm_mode = os.getenv("LITELLM_MODE", "DEV")  # "PRODUCTION", "DEV"
if litellm_mode == "DEV":
    dotenv.load_dotenv()
#############################################
if set_verbose == True:
    _turn_on_debug()
#############################################
### Callbacks /Logging / Success / Failure Handlers ###
input_callback: List[Union[str, Callable]] = []
success_callback: List[Union[str, Callable]] = []
failure_callback: List[Union[str, Callable]] = []
service_callback: List[Union[str, Callable]] = []
_custom_logger_compatible_callbacks_literal = Literal[
    "lago",
    "openmeter",
    "logfire",
    "dynamic_rate_limiter",
    "langsmith",
    "prometheus",
    "datadog",
    "galileo",
    "braintrust",
    "arize",
    "gcs_bucket",
]
_known_custom_logger_compatible_callbacks: List = list(
    get_args(_custom_logger_compatible_callbacks_literal)
)
callbacks: List[Union[Callable, _custom_logger_compatible_callbacks_literal]] = []
langfuse_default_tags: Optional[List[str]] = None
langsmith_batch_size: Optional[int] = None
_async_input_callback: List[Callable] = (
    []
)  # internal variable - async custom callbacks are routed here.
_async_success_callback: List[Union[str, Callable]] = (
    []
)  # internal variable - async custom callbacks are routed here.
_async_failure_callback: List[Callable] = (
    []
)  # internal variable - async custom callbacks are routed here.
pre_call_rules: List[Callable] = []
post_call_rules: List[Callable] = []
turn_off_message_logging: Optional[bool] = False
log_raw_request_response: bool = False
redact_messages_in_exceptions: Optional[bool] = False
redact_user_api_key_info: Optional[bool] = False
store_audit_logs = False  # Enterprise feature, allow users to see audit logs
## end of callbacks #############

email: Optional[str] = (
    None  # Not used anymore, will be removed in next MAJOR release - https://github.com/BerriAI/litellm/discussions/648
)
token: Optional[str] = (
    None  # Not used anymore, will be removed in next MAJOR release - https://github.com/BerriAI/litellm/discussions/648
)
telemetry = True
max_tokens = 256  # OpenAI Defaults
drop_params = bool(os.getenv("LITELLM_DROP_PARAMS", False))
modify_params = False
retry = True
### AUTH ###
api_key: Optional[str] = None
openai_key: Optional[str] = None
databricks_key: Optional[str] = None
azure_key: Optional[str] = None
anthropic_key: Optional[str] = None
replicate_key: Optional[str] = None
cohere_key: Optional[str] = None
clarifai_key: Optional[str] = None
maritalk_key: Optional[str] = None
ai21_key: Optional[str] = None
ollama_key: Optional[str] = None
openrouter_key: Optional[str] = None
predibase_key: Optional[str] = None
huggingface_key: Optional[str] = None
vertex_project: Optional[str] = None
vertex_location: Optional[str] = None
predibase_tenant_id: Optional[str] = None
togetherai_api_key: Optional[str] = None
cloudflare_api_key: Optional[str] = None
baseten_key: Optional[str] = None
aleph_alpha_key: Optional[str] = None
nlp_cloud_key: Optional[str] = None
common_cloud_provider_auth_params: dict = {
    "params": ["project", "region_name", "token"],
    "providers": ["vertex_ai", "bedrock", "watsonx", "azure", "vertex_ai_beta"],
}
use_client: bool = False
ssl_verify: Union[str, bool] = True
ssl_certificate: Optional[str] = None
disable_streaming_logging: bool = False
in_memory_llm_clients_cache: dict = {}
safe_memory_mode: bool = False
enable_azure_ad_token_refresh: Optional[bool] = False
### DEFAULT AZURE API VERSION ###
AZURE_DEFAULT_API_VERSION = "2024-08-01-preview"  # this is updated to the latest
### COHERE EMBEDDINGS DEFAULT TYPE ###
COHERE_DEFAULT_EMBEDDING_INPUT_TYPE = "search_document"
### GUARDRAILS ###
llamaguard_model_name: Optional[str] = None
openai_moderations_model_name: Optional[str] = None
presidio_ad_hoc_recognizers: Optional[str] = None
google_moderation_confidence_threshold: Optional[float] = None
llamaguard_unsafe_content_categories: Optional[str] = None
blocked_user_list: Optional[Union[str, List]] = None
banned_keywords_list: Optional[Union[str, List]] = None
llm_guard_mode: Literal["all", "key-specific", "request-specific"] = "all"
guardrail_name_config_map: Dict[str, GuardrailItem] = {}
##################
### PREVIEW FEATURES ###
enable_preview_features: bool = False
return_response_headers: bool = (
    False  # get response headers from LLM Api providers - example x-remaining-requests,
)
enable_json_schema_validation: bool = False
##################
logging: bool = True
enable_loadbalancing_on_batch_endpoints: Optional[bool] = None
enable_caching_on_provider_specific_optional_params: bool = (
    False  # feature-flag for caching on optional params - e.g. 'top_k'
)
caching: bool = (
    False  # Not used anymore, will be removed in next MAJOR release - https://github.com/BerriAI/litellm/discussions/648
)
always_read_redis: bool = (
    True  # always use redis for rate limiting logic on litellm proxy
)
caching_with_models: bool = (
    False  # # Not used anymore, will be removed in next MAJOR release - https://github.com/BerriAI/litellm/discussions/648
)
cache: Optional[Cache] = (
    None  # cache object <- use this - https://docs.litellm.ai/docs/caching
)
default_in_memory_ttl: Optional[float] = None
default_redis_ttl: Optional[float] = None
model_alias_map: Dict[str, str] = {}
model_group_alias_map: Dict[str, str] = {}
max_budget: float = 0.0  # set the max budget across all providers
budget_duration: Optional[str] = (
    None  # proxy only - resets budget after fixed duration. You can set duration as seconds ("30s"), minutes ("30m"), hours ("30h"), days ("30d").
)
default_soft_budget: float = (
    50.0  # by default all litellm proxy keys have a soft budget of 50.0
)
forward_traceparent_to_llm_provider: bool = False
_openai_finish_reasons = ["stop", "length", "function_call", "content_filter", "null"]
_openai_completion_params = [
    "functions",
    "function_call",
    "temperature",
    "temperature",
    "top_p",
    "n",
    "stream",
    "stop",
    "max_tokens",
    "presence_penalty",
    "frequency_penalty",
    "logit_bias",
    "user",
    "request_timeout",
    "api_base",
    "api_version",
    "api_key",
    "deployment_id",
    "organization",
    "base_url",
    "default_headers",
    "timeout",
    "response_format",
    "seed",
    "tools",
    "tool_choice",
    "max_retries",
]
_litellm_completion_params = [
    "metadata",
    "acompletion",
    "caching",
    "mock_response",
    "api_key",
    "api_version",
    "api_base",
    "force_timeout",
    "logger_fn",
    "verbose",
    "custom_llm_provider",
    "litellm_logging_obj",
    "litellm_call_id",
    "use_client",
    "id",
    "fallbacks",
    "azure",
    "headers",
    "model_list",
    "num_retries",
    "context_window_fallback_dict",
    "roles",
    "final_prompt_value",
    "bos_token",
    "eos_token",
    "request_timeout",
    "complete_response",
    "self",
    "client",
    "rpm",
    "tpm",
    "input_cost_per_token",
    "output_cost_per_token",
    "hf_model_name",
    "model_info",
    "proxy_server_request",
    "preset_cache_key",
]
_current_cost = 0  # private variable, used if max budget is set
error_logs: Dict = {}
add_function_to_prompt: bool = (
    False  # if function calling not supported by api, append function call details to system prompt
)
client_session: Optional[httpx.Client] = None
aclient_session: Optional[httpx.AsyncClient] = None
model_fallbacks: Optional[List] = None  # Deprecated for 'litellm.fallbacks'
model_cost_map_url: str = (
    "https://raw.githubusercontent.com/BerriAI/litellm/main/model_prices_and_context_window.json"
)
suppress_debug_info = False
dynamodb_table_name: Optional[str] = None
s3_callback_params: Optional[Dict] = None
generic_logger_headers: Optional[Dict] = None
default_key_generate_params: Optional[Dict] = None
upperbound_key_generate_params: Optional[LiteLLM_UpperboundKeyGenerateParams] = None
default_internal_user_params: Optional[Dict] = None
default_team_settings: Optional[List] = None
max_user_budget: Optional[float] = None
default_max_internal_user_budget: Optional[float] = None
max_internal_user_budget: Optional[float] = None
internal_user_budget_duration: Optional[str] = None
max_end_user_budget: Optional[float] = None
#### REQUEST PRIORITIZATION ####
priority_reservation: Optional[Dict[str, float]] = None
#### RELIABILITY ####
REPEATED_STREAMING_CHUNK_LIMIT = 100  # catch if model starts looping the same chunk while streaming. Uses high default to prevent false positives.
request_timeout: float = 6000
module_level_aclient = AsyncHTTPHandler(timeout=request_timeout)
module_level_client = HTTPHandler(timeout=request_timeout)
num_retries: Optional[int] = None  # per model endpoint
default_fallbacks: Optional[List] = None
fallbacks: Optional[List] = None
context_window_fallbacks: Optional[List] = None
content_policy_fallbacks: Optional[List] = None
allowed_fails: int = 3
num_retries_per_request: Optional[int] = (
    None  # for the request overall (incl. fallbacks + model retries)
)
####### SECRET MANAGERS #####################
secret_manager_client: Optional[Any] = (
    None  # list of instantiated key management clients - e.g. azure kv, infisical, etc.
)
_google_kms_resource_name: Optional[str] = None
_key_management_system: Optional[KeyManagementSystem] = None
_key_management_settings: Optional[KeyManagementSettings] = None
#### PII MASKING ####
output_parse_pii: bool = False
#############################################


def get_model_cost_map(url: str):
    json_text = """{
    "sample_spec": {
        "max_tokens": "set to max_output_tokens if provider specifies it. IF not set to max_tokens provider specifies", 
        "max_input_tokens": "max input tokens, if the provider specifies it. if not default to max_tokens",
        "max_output_tokens": "max output tokens, if the provider specifies it. if not default to max_tokens", 
        "input_cost_per_token": 0.0000,
        "output_cost_per_token": 0.000,
        "litellm_provider": "one of https://docs.litellm.ai/docs/providers",
        "mode": "one of chat, embedding, completion, image_generation, audio_transcription",
        "supports_function_calling": true,
        "supports_parallel_function_calling": true,
        "supports_vision": true
    },
    "gpt-4": {
        "max_tokens": 4096, 
        "max_input_tokens": 8192,
        "max_output_tokens": 4096, 
        "input_cost_per_token": 0.00003,
        "output_cost_per_token": 0.00006,
        "litellm_provider": "openai",
        "mode": "chat",
        "supports_function_calling": true
    },
    "gpt-4o": {
        "max_tokens": 4096,
        "max_input_tokens": 128000,
        "max_output_tokens": 4096,
        "input_cost_per_token": 0.000005,
        "output_cost_per_token": 0.000015,
        "litellm_provider": "openai",
        "mode": "chat",
        "supports_function_calling": true,
        "supports_parallel_function_calling": true,
        "supports_vision": true
    },
    "gpt-4o-mini": {
        "max_tokens": 16384,
        "max_input_tokens": 128000,
        "max_output_tokens": 16384,
        "input_cost_per_token": 0.00000015,
        "output_cost_per_token": 0.00000060,
        "litellm_provider": "openai",
        "mode": "chat",
        "supports_function_calling": true,
        "supports_parallel_function_calling": true,
        "supports_vision": true
    },
    "gpt-4o-mini-2024-07-18": {
        "max_tokens": 16384,
        "max_input_tokens": 128000,
        "max_output_tokens": 16384,
        "input_cost_per_token": 0.00000015,
        "output_cost_per_token": 0.00000060,
        "litellm_provider": "openai",
        "mode": "chat",
        "supports_function_calling": true,
        "supports_parallel_function_calling": true,
        "supports_vision": true
    },
    "chatgpt-4o-latest": {
        "max_tokens": 4096,
        "max_input_tokens": 128000,
        "max_output_tokens": 4096,
        "input_cost_per_token": 0.000005,
        "output_cost_per_token": 0.000015,
        "litellm_provider": "openai",
        "mode": "chat",
        "supports_function_calling": true,
        "supports_parallel_function_calling": true,
        "supports_vision": true
    },
    "gpt-4o-2024-05-13": {
        "max_tokens": 4096,
        "max_input_tokens": 128000,
        "max_output_tokens": 4096,
        "input_cost_per_token": 0.000005,
        "output_cost_per_token": 0.000015,
        "litellm_provider": "openai",
        "mode": "chat",
        "supports_function_calling": true,
        "supports_parallel_function_calling": true,
        "supports_vision": true
    },
    "gpt-4o-2024-02-15-preview": {
        "max_tokens": 16384,
        "max_input_tokens": 128000,
        "max_output_tokens": 16384,
        "input_cost_per_token": 0.0000025,
        "output_cost_per_token": 0.000010,
        "litellm_provider": "openai",
        "mode": "chat",
        "supports_function_calling": true,
        "supports_parallel_function_calling": true,
        "supports_vision": true
    },
    "gpt-4o-2024-08-06": {
        "max_tokens": 16384,
        "max_input_tokens": 128000,
        "max_output_tokens": 16384,
        "input_cost_per_token": 0.0000025,
        "output_cost_per_token": 0.000010,
        "litellm_provider": "openai",
        "mode": "chat",
        "supports_function_calling": true,
        "supports_parallel_function_calling": true,
        "supports_vision": true
    },
    "gpt-4-turbo-preview": {
        "max_tokens": 4096,
        "max_input_tokens": 128000,
        "max_output_tokens": 4096,
        "input_cost_per_token": 0.00001,
        "output_cost_per_token": 0.00003,
        "litellm_provider": "openai",
        "mode": "chat",
        "supports_function_calling": true,
        "supports_parallel_function_calling": true
    },
    "gpt-4-0314": {
        "max_tokens": 4096,
        "max_input_tokens": 8192,
        "max_output_tokens": 4096,
        "input_cost_per_token": 0.00003,
        "output_cost_per_token": 0.00006,
        "litellm_provider": "openai",
        "mode": "chat"
    },
    "gpt-4-0613": {
        "max_tokens": 4096,
        "max_input_tokens": 8192,
        "max_output_tokens": 4096,
        "input_cost_per_token": 0.00003,
        "output_cost_per_token": 0.00006,
        "litellm_provider": "openai",
        "mode": "chat",
        "supports_function_calling": true
    },
    "gpt-4-32k": {
        "max_tokens": 4096,
        "max_input_tokens": 32768,
        "max_output_tokens": 4096,
        "input_cost_per_token": 0.00006,
        "output_cost_per_token": 0.00012,
        "litellm_provider": "openai",
        "mode": "chat"
    },
    "gpt-4-32k-0314": {
        "max_tokens": 4096,
        "max_input_tokens": 32768,
        "max_output_tokens": 4096,
        "input_cost_per_token": 0.00006,
        "output_cost_per_token": 0.00012,
        "litellm_provider": "openai",
        "mode": "chat"
    },
    "gpt-4-32k-0613": {
        "max_tokens": 4096,
        "max_input_tokens": 32768,
        "max_output_tokens": 4096,
        "input_cost_per_token": 0.00006,
        "output_cost_per_token": 0.00012,
        "litellm_provider": "openai",
        "mode": "chat"
    },
    "gpt-4-turbo": {
        "max_tokens": 4096,
        "max_input_tokens": 128000,
        "max_output_tokens": 4096,
        "input_cost_per_token": 0.00001,
        "output_cost_per_token": 0.00003,
        "litellm_provider": "openai",
        "mode": "chat",
        "supports_function_calling": true,
        "supports_parallel_function_calling": true,
        "supports_vision": true
    },
    "gpt-4-turbo-2024-04-09": {
        "max_tokens": 4096,
        "max_input_tokens": 128000,
        "max_output_tokens": 4096,
        "input_cost_per_token": 0.00001,
        "output_cost_per_token": 0.00003,
        "litellm_provider": "openai",
        "mode": "chat",
        "supports_function_calling": true,
        "supports_parallel_function_calling": true,
        "supports_vision": true
    },
    "gpt-4-1106-preview": {
        "max_tokens": 4096,
        "max_input_tokens": 128000,
        "max_output_tokens": 4096,
        "input_cost_per_token": 0.00001,
        "output_cost_per_token": 0.00003,
        "litellm_provider": "openai",
        "mode": "chat",
        "supports_function_calling": true,
        "supports_parallel_function_calling": true
    },
    "gpt-4-0125-preview": {
        "max_tokens": 4096,
        "max_input_tokens": 128000,
        "max_output_tokens": 4096,
        "input_cost_per_token": 0.00001,
        "output_cost_per_token": 0.00003,
        "litellm_provider": "openai",
        "mode": "chat",
        "supports_function_calling": true,
        "supports_parallel_function_calling": true
    },
    "gpt-4-vision-preview": {
        "max_tokens": 4096,
        "max_input_tokens": 128000,
        "max_output_tokens": 4096,
        "input_cost_per_token": 0.00001,
        "output_cost_per_token": 0.00003,
        "litellm_provider": "openai",
        "mode": "chat",
        "supports_vision": true
    },
    "gpt-4-1106-vision-preview": {
        "max_tokens": 4096,
        "max_input_tokens": 128000,
        "max_output_tokens": 4096,
        "input_cost_per_token": 0.00001,
        "output_cost_per_token": 0.00003,
        "litellm_provider": "openai",
        "mode": "chat",
        "supports_vision": true
    },
    "gpt-3.5-turbo": {
        "max_tokens": 4097,
        "max_input_tokens": 16385,
        "max_output_tokens": 4096,
        "input_cost_per_token": 0.0000015,
        "output_cost_per_token": 0.000002,
        "litellm_provider": "openai",
        "mode": "chat",
        "supports_function_calling": true
    },
    "gpt-3.5-turbo-0301": {
        "max_tokens": 4097,
        "max_input_tokens": 4097,
        "max_output_tokens": 4096,
        "input_cost_per_token": 0.0000015,
        "output_cost_per_token": 0.000002,
        "litellm_provider": "openai",
        "mode": "chat"
    },
    "gpt-3.5-turbo-0613": {
        "max_tokens": 4097,
        "max_input_tokens": 4097,
        "max_output_tokens": 4096,
        "input_cost_per_token": 0.0000015,
        "output_cost_per_token": 0.000002,
        "litellm_provider": "openai",
        "mode": "chat",
        "supports_function_calling": true
    },
    "gpt-3.5-turbo-1106": {
        "max_tokens": 16385,
        "max_input_tokens": 16385,
        "max_output_tokens": 4096,
        "input_cost_per_token": 0.0000010,
        "output_cost_per_token": 0.0000020,
        "litellm_provider": "openai",
        "mode": "chat",
        "supports_function_calling": true,
        "supports_parallel_function_calling": true
    },
    "gpt-3.5-turbo-0125": {
        "max_tokens": 16385,
        "max_input_tokens": 16385,
        "max_output_tokens": 4096,
        "input_cost_per_token": 0.0000005,
        "output_cost_per_token": 0.0000015,
        "litellm_provider": "openai",
        "mode": "chat",
        "supports_function_calling": true,
        "supports_parallel_function_calling": true
    },
    "gpt-3.5-turbo-16k": {
        "max_tokens": 16385,
        "max_input_tokens": 16385,
        "max_output_tokens": 4096,
        "input_cost_per_token": 0.000003,
        "output_cost_per_token": 0.000004,
        "litellm_provider": "openai",
        "mode": "chat"
    },
    "gpt-3.5-turbo-16k-0613": {
        "max_tokens": 16385,
        "max_input_tokens": 16385,
        "max_output_tokens": 4096,
        "input_cost_per_token": 0.000003,
        "output_cost_per_token": 0.000004,
        "litellm_provider": "openai",
        "mode": "chat"
    },
    "ft:gpt-3.5-turbo": {
        "max_tokens": 4096,
        "max_input_tokens": 16385,
        "max_output_tokens": 4096,
        "input_cost_per_token": 0.000003,
        "output_cost_per_token": 0.000006,
        "litellm_provider": "openai",
        "mode": "chat"
    },
    "ft:gpt-3.5-turbo-0125": {
        "max_tokens": 4096,
        "max_input_tokens": 16385,
        "max_output_tokens": 4096,
        "input_cost_per_token": 0.000003,
        "output_cost_per_token": 0.000006,
        "litellm_provider": "openai",
        "mode": "chat"
    },
    "ft:gpt-3.5-turbo-1106": {
        "max_tokens": 4096,
        "max_input_tokens": 16385,
        "max_output_tokens": 4096,
        "input_cost_per_token": 0.000003,
        "output_cost_per_token": 0.000006,
        "litellm_provider": "openai",
        "mode": "chat"
    },
    "ft:gpt-3.5-turbo-0613": {
        "max_tokens": 4096,
        "max_input_tokens": 4096,
        "max_output_tokens": 4096,
        "input_cost_per_token": 0.000003,
        "output_cost_per_token": 0.000006,
        "litellm_provider": "openai",
        "mode": "chat"
    },
    "ft:gpt-4-0613": {
        "max_tokens": 4096,
        "max_input_tokens": 8192,
        "max_output_tokens": 4096,
        "input_cost_per_token": 0.00003,
        "output_cost_per_token": 0.00006,
        "litellm_provider": "openai",
        "mode": "chat",
        "supports_function_calling": true,
        "source": "OpenAI needs to add pricing for this ft model, will be updated when added by OpenAI. Defaulting to base model pricing"
    },
    "ft:gpt-4o-2024-08-06": {
        "max_tokens": 16384,
        "max_input_tokens": 128000,
        "max_output_tokens": 16384,
        "input_cost_per_token": 0.00000375,
        "output_cost_per_token": 0.000015,
        "litellm_provider": "openai",
        "mode": "chat",
        "supports_function_calling": true,
        "supports_parallel_function_calling": true,
        "supports_vision": true
    },
    "ft:gpt-4o-mini-2024-07-18": {
        "max_tokens": 16384,
        "max_input_tokens": 128000,
        "max_output_tokens": 16384,
        "input_cost_per_token": 0.0000003,
        "output_cost_per_token": 0.0000012,
        "litellm_provider": "openai",
        "mode": "chat",
        "supports_function_calling": true,
        "supports_parallel_function_calling": true,
        "supports_vision": true
    },
    "ft:davinci-002": {
        "max_tokens": 16384,
        "max_input_tokens": 16384,
        "max_output_tokens": 4096,
        "input_cost_per_token": 0.000002,
        "output_cost_per_token": 0.000002,
        "litellm_provider": "text-completion-openai",
        "mode": "completion"
    },
    "ft:babbage-002": {
        "max_tokens": 16384,
        "max_input_tokens": 16384,
        "max_output_tokens": 4096,
        "input_cost_per_token": 0.0000004,
        "output_cost_per_token": 0.0000004,
        "litellm_provider": "text-completion-openai",
        "mode": "completion"
    },
    "text-embedding-3-large": {
        "max_tokens": 8191,
        "max_input_tokens": 8191,
        "output_vector_size": 3072,
        "input_cost_per_token": 0.00000013,
        "output_cost_per_token": 0.000000,
        "litellm_provider": "openai",
        "mode": "embedding"
    },
    "text-embedding-3-small": {
        "max_tokens": 8191,
        "max_input_tokens": 8191,
        "output_vector_size": 1536, 
        "input_cost_per_token": 0.00000002,
        "output_cost_per_token": 0.000000,
        "litellm_provider": "openai",
        "mode": "embedding"
    },
    "text-embedding-ada-002": {
        "max_tokens": 8191,
        "max_input_tokens": 8191,
        "output_vector_size": 1536, 
        "input_cost_per_token": 0.0000001,
        "output_cost_per_token": 0.000000,
        "litellm_provider": "openai",
        "mode": "embedding"
    },
    "text-embedding-ada-002-v2": {
        "max_tokens": 8191,
        "max_input_tokens": 8191,
        "input_cost_per_token": 0.0000001,
        "output_cost_per_token": 0.000000,
        "litellm_provider": "openai",
        "mode": "embedding"
    },
    "text-moderation-stable": {
        "max_tokens": 32768,
        "max_input_tokens": 32768,
        "max_output_tokens": 0,
        "input_cost_per_token": 0.000000,
        "output_cost_per_token": 0.000000,
        "litellm_provider": "openai",
        "mode": "moderations"
    },
    "text-moderation-007": {
        "max_tokens": 32768,
        "max_input_tokens": 32768,
        "max_output_tokens": 0,
        "input_cost_per_token": 0.000000,
        "output_cost_per_token": 0.000000,
        "litellm_provider": "openai",
        "mode": "moderations"
    },
    "text-moderation-latest": {
        "max_tokens": 32768,
        "max_input_tokens": 32768,
        "max_output_tokens": 0,
        "input_cost_per_token": 0.000000,
        "output_cost_per_token": 0.000000,
        "litellm_provider": "openai",
        "mode": "moderations"
    },
    "256-x-256/dall-e-2": {
        "mode": "image_generation",
        "input_cost_per_pixel": 0.00000024414,
        "output_cost_per_pixel": 0.0,
        "litellm_provider": "openai"
    },
    "512-x-512/dall-e-2": {
        "mode": "image_generation",
        "input_cost_per_pixel": 0.0000000686,
        "output_cost_per_pixel": 0.0,
        "litellm_provider": "openai"
    },
    "1024-x-1024/dall-e-2": {
        "mode": "image_generation",
        "input_cost_per_pixel": 0.000000019,
        "output_cost_per_pixel": 0.0,
        "litellm_provider": "openai"
    },
    "hd/1024-x-1792/dall-e-3": {
        "mode": "image_generation",
        "input_cost_per_pixel": 0.00000006539,
        "output_cost_per_pixel": 0.0,
        "litellm_provider": "openai"
    },
    "hd/1792-x-1024/dall-e-3": {
        "mode": "image_generation",
        "input_cost_per_pixel": 0.00000006539,
        "output_cost_per_pixel": 0.0,
        "litellm_provider": "openai"
    },
    "hd/1024-x-1024/dall-e-3": {
        "mode": "image_generation",
        "input_cost_per_pixel": 0.00000007629,
        "output_cost_per_pixel": 0.0,
        "litellm_provider": "openai"
    },
    "standard/1024-x-1792/dall-e-3": {
        "mode": "image_generation",
        "input_cost_per_pixel": 0.00000004359,
        "output_cost_per_pixel": 0.0,
        "litellm_provider": "openai"
    },
    "standard/1792-x-1024/dall-e-3": {
        "mode": "image_generation",
        "input_cost_per_pixel": 0.00000004359,
        "output_cost_per_pixel": 0.0,
        "litellm_provider": "openai"
    },
    "standard/1024-x-1024/dall-e-3": {
        "mode": "image_generation",
        "input_cost_per_pixel": 0.0000000381469,
        "output_cost_per_pixel": 0.0,
        "litellm_provider": "openai"
    },
    "whisper-1": {
        "mode": "audio_transcription",
        "input_cost_per_second": 0,
        "output_cost_per_second": 0.0001, 
        "litellm_provider": "openai"
    }, 
    "tts-1": {
        "mode": "audio_speech", 
        "input_cost_per_character": 0.000015,
        "litellm_provider": "openai"
    },
    "tts-1-hd": {
        "mode": "audio_speech", 
        "input_cost_per_character": 0.000030,
        "litellm_provider": "openai"
    },
    "azure/tts-1": {
        "mode": "audio_speech", 
        "input_cost_per_character": 0.000015,
        "litellm_provider": "azure"
    },
    "azure/tts-1-hd": {
        "mode": "audio_speech", 
        "input_cost_per_character": 0.000030,
        "litellm_provider": "azure"
    },
    "azure/whisper-1": {
        "mode": "audio_transcription",
        "input_cost_per_second": 0, 
        "output_cost_per_second": 0.0001, 
        "litellm_provider": "azure"
    },
    "azure/gpt-4o": {
        "max_tokens": 4096,
        "max_input_tokens": 128000,
        "max_output_tokens": 4096,
        "input_cost_per_token": 0.000005,
        "output_cost_per_token": 0.000015,
        "litellm_provider": "azure",
        "mode": "chat",
        "supports_function_calling": true,
        "supports_parallel_function_calling": true,
        "supports_vision": true
    },
    "azure/gpt-4o-2024-08-06": {
        "max_tokens": 16384,
        "max_input_tokens": 128000,
        "max_output_tokens": 16384,
        "input_cost_per_token": 0.00000275,
        "output_cost_per_token": 0.000011,
        "litellm_provider": "azure",
        "mode": "chat",
        "supports_function_calling": true,
        "supports_parallel_function_calling": true,
        "supports_vision": true
    },
    "azure/global-standard/gpt-4o-2024-08-06": {
        "max_tokens": 16384,
        "max_input_tokens": 128000,
        "max_output_tokens": 16384,
        "input_cost_per_token": 0.0000025,
        "output_cost_per_token": 0.000010,
        "litellm_provider": "azure",
        "mode": "chat",
        "supports_function_calling": true,
        "supports_parallel_function_calling": true,
        "supports_vision": true
    },
    "azure/global-standard/gpt-4o-mini": {
        "max_tokens": 16384,
        "max_input_tokens": 128000,
        "max_output_tokens": 16384,
        "input_cost_per_token": 0.00000015,
        "output_cost_per_token": 0.00000060,
        "litellm_provider": "azure",
        "mode": "chat",
        "supports_function_calling": true,
        "supports_parallel_function_calling": true,
        "supports_vision": true
    },
    "azure/gpt-4o-mini": {
        "max_tokens": 16384,
        "max_input_tokens": 128000,
        "max_output_tokens": 16384,
        "input_cost_per_token": 0.000000165,
        "output_cost_per_token": 0.00000066,
        "litellm_provider": "azure",
        "mode": "chat",
        "supports_function_calling": true,
        "supports_parallel_function_calling": true,
        "supports_vision": true
    },
    "azure/gpt-4-turbo-2024-04-09": {
        "max_tokens": 4096,
        "max_input_tokens": 128000,
        "max_output_tokens": 4096,
        "input_cost_per_token": 0.00001,
        "output_cost_per_token": 0.00003,
        "litellm_provider": "azure",
        "mode": "chat",
        "supports_function_calling": true,
        "supports_parallel_function_calling": true,
        "supports_vision": true
    },
    "azure/gpt-4-0125-preview": {
        "max_tokens": 4096,
        "max_input_tokens": 128000,
        "max_output_tokens": 4096,
        "input_cost_per_token": 0.00001,
        "output_cost_per_token": 0.00003,
        "litellm_provider": "azure",
        "mode": "chat",
        "supports_function_calling": true,
        "supports_parallel_function_calling": true
    },
    "azure/gpt-4-1106-preview": {
        "max_tokens": 4096,
        "max_input_tokens": 128000,
        "max_output_tokens": 4096,
        "input_cost_per_token": 0.00001,
        "output_cost_per_token": 0.00003,
        "litellm_provider": "azure",
        "mode": "chat",
        "supports_function_calling": true,
        "supports_parallel_function_calling": true
    },
    "azure/gpt-4-0613": {
        "max_tokens": 4096,
        "max_input_tokens": 8192,
        "max_output_tokens": 4096,
        "input_cost_per_token": 0.00003,
        "output_cost_per_token": 0.00006,
        "litellm_provider": "azure",
        "mode": "chat",
        "supports_function_calling": true
    },
    "azure/gpt-4-32k-0613": {
        "max_tokens": 4096,
        "max_input_tokens": 32768,
        "max_output_tokens": 4096,
        "input_cost_per_token": 0.00006,
        "output_cost_per_token": 0.00012,
        "litellm_provider": "azure",
        "mode": "chat"
    },
    "azure/gpt-4-32k": {
        "max_tokens": 4096,
        "max_input_tokens": 32768,
        "max_output_tokens": 4096,
        "input_cost_per_token": 0.00006,
        "output_cost_per_token": 0.00012,
        "litellm_provider": "azure",
        "mode": "chat"
    },
    "azure/gpt-4": {
        "max_tokens": 4096,
        "max_input_tokens": 8192,
        "max_output_tokens": 4096,
        "input_cost_per_token": 0.00003,
        "output_cost_per_token": 0.00006,
        "litellm_provider": "azure",
        "mode": "chat",
        "supports_function_calling": true
    },
    "azure/gpt-4-turbo": {
        "max_tokens": 4096,
        "max_input_tokens": 128000,
        "max_output_tokens": 4096,
        "input_cost_per_token": 0.00001,
        "output_cost_per_token": 0.00003,
        "litellm_provider": "azure", 
        "mode": "chat",
        "supports_function_calling": true,
        "supports_parallel_function_calling": true
    },
    "azure/gpt-4-turbo-vision-preview": {
        "max_tokens": 4096,
        "max_input_tokens": 128000,
        "max_output_tokens": 4096,
        "input_cost_per_token": 0.00001,
        "output_cost_per_token": 0.00003,
        "litellm_provider": "azure", 
        "mode": "chat",
        "supports_vision": true
    },
    "azure/gpt-35-turbo-16k-0613": {
        "max_tokens": 4096,
        "max_input_tokens": 16385,
        "max_output_tokens": 4096,
        "input_cost_per_token": 0.000003,
        "output_cost_per_token": 0.000004,
        "litellm_provider": "azure",
        "mode": "chat",
        "supports_function_calling": true
    },
    "azure/gpt-35-turbo-1106": {
        "max_tokens": 4096,
        "max_input_tokens": 16384,
        "max_output_tokens": 4096,
        "input_cost_per_token": 0.000001,
        "output_cost_per_token": 0.000002,
        "litellm_provider": "azure",
        "mode": "chat",
        "supports_function_calling": true,
        "supports_parallel_function_calling": true
    },
    "azure/gpt-35-turbo-0125": {
        "max_tokens": 4096,
        "max_input_tokens": 16384,
        "max_output_tokens": 4096,
        "input_cost_per_token": 0.0000005,
        "output_cost_per_token": 0.0000015,
        "litellm_provider": "azure",
        "mode": "chat",
        "supports_function_calling": true,
        "supports_parallel_function_calling": true
    },
    "azure/gpt-35-turbo-16k": {
        "max_tokens": 4096,
        "max_input_tokens": 16385,
        "max_output_tokens": 4096,
        "input_cost_per_token": 0.000003,
        "output_cost_per_token": 0.000004,
        "litellm_provider": "azure",
        "mode": "chat"
    },
    "azure/gpt-35-turbo": {
        "max_tokens": 4096,
        "max_input_tokens": 4097,
        "max_output_tokens": 4096,
        "input_cost_per_token": 0.0000005,
        "output_cost_per_token": 0.0000015,
        "litellm_provider": "azure",
        "mode": "chat",
        "supports_function_calling": true
    },
    "azure/gpt-3.5-turbo-instruct-0914": {
        "max_tokens": 4097,
        "max_input_tokens": 4097,
        "input_cost_per_token": 0.0000015,
        "output_cost_per_token": 0.000002,
        "litellm_provider": "text-completion-openai",
        "mode": "completion"
    },
    "azure/gpt-35-turbo-instruct": {
        "max_tokens": 4097,
        "max_input_tokens": 4097,
        "input_cost_per_token": 0.0000015,
        "output_cost_per_token": 0.000002,
        "litellm_provider": "text-completion-openai",
        "mode": "completion"
    },
    "azure/mistral-large-latest": {
        "max_tokens": 32000,
        "max_input_tokens": 32000,
        "input_cost_per_token": 0.000008,
        "output_cost_per_token": 0.000024,
        "litellm_provider": "azure",
        "mode": "chat",
        "supports_function_calling": true
    },
    "azure/mistral-large-2402": {
        "max_tokens": 32000,
        "max_input_tokens": 32000,
        "input_cost_per_token": 0.000008,
        "output_cost_per_token": 0.000024,
        "litellm_provider": "azure",
        "mode": "chat",
        "supports_function_calling": true
    },
    "azure/command-r-plus": {
        "max_tokens": 4096, 
        "max_input_tokens": 128000,
        "max_output_tokens": 4096,
        "input_cost_per_token": 0.000003,
        "output_cost_per_token": 0.000015,
        "litellm_provider": "azure",
        "mode": "chat",
        "supports_function_calling": true
    },
    "azure/ada": {
        "max_tokens": 8191,
        "max_input_tokens": 8191,
        "input_cost_per_token": 0.0000001,
        "output_cost_per_token": 0.000000,
        "litellm_provider": "azure",
        "mode": "embedding"
    },
    "azure/text-embedding-ada-002": {
        "max_tokens": 8191,
        "max_input_tokens": 8191,
        "input_cost_per_token": 0.0000001,
        "output_cost_per_token": 0.000000,
        "litellm_provider": "azure",
        "mode": "embedding"
    },
    "azure/text-embedding-3-large": {
        "max_tokens": 8191,
        "max_input_tokens": 8191,
        "input_cost_per_token": 0.00000013,
        "output_cost_per_token": 0.000000,
        "litellm_provider": "azure",
        "mode": "embedding"
    },
    "azure/text-embedding-3-small": {
        "max_tokens": 8191,
        "max_input_tokens": 8191,
        "input_cost_per_token": 0.00000002,
        "output_cost_per_token": 0.000000,
        "litellm_provider": "azure",
        "mode": "embedding"
    },    
    "azure/standard/1024-x-1024/dall-e-3": {
        "input_cost_per_pixel": 0.0000000381469,
        "output_cost_per_token": 0.0,
        "litellm_provider": "azure", 
        "mode": "image_generation"
    },
    "azure/hd/1024-x-1024/dall-e-3": {
        "input_cost_per_pixel": 0.00000007629,
        "output_cost_per_token": 0.0,
        "litellm_provider": "azure", 
        "mode": "image_generation"
    },
    "azure/standard/1024-x-1792/dall-e-3": {
        "input_cost_per_pixel": 0.00000004359,
        "output_cost_per_token": 0.0,
        "litellm_provider": "azure", 
        "mode": "image_generation"
    },
    "azure/standard/1792-x-1024/dall-e-3": {
        "input_cost_per_pixel": 0.00000004359,
        "output_cost_per_token": 0.0,
        "litellm_provider": "azure", 
        "mode": "image_generation"
    },
    "azure/hd/1024-x-1792/dall-e-3": {
        "input_cost_per_pixel": 0.00000006539,
        "output_cost_per_token": 0.0,
        "litellm_provider": "azure", 
        "mode": "image_generation"
    },
    "azure/hd/1792-x-1024/dall-e-3": {
        "input_cost_per_pixel": 0.00000006539,
        "output_cost_per_token": 0.0,
        "litellm_provider": "azure", 
        "mode": "image_generation"
    },
    "azure/standard/1024-x-1024/dall-e-2": {
        "input_cost_per_pixel": 0.0,
        "output_cost_per_token": 0.0,
        "litellm_provider": "azure", 
        "mode": "image_generation"
    },
    "azure_ai/jamba-instruct": {
        "max_tokens": 4096,
        "max_input_tokens": 70000,
        "max_output_tokens": 4096,
        "input_cost_per_token": 0.0000005,
        "output_cost_per_token": 0.0000007,
        "litellm_provider": "azure_ai",
        "mode": "chat"
    },
    "azure_ai/mistral-large": {
        "max_tokens": 8191,
        "max_input_tokens": 32000,
        "max_output_tokens": 8191,
        "input_cost_per_token": 0.000004,
        "output_cost_per_token": 0.000012,
        "litellm_provider": "azure_ai",
        "mode": "chat",
        "supports_function_calling": true
    },
    "azure_ai/mistral-small": {
        "max_tokens": 8191,
        "max_input_tokens": 32000,
        "max_output_tokens": 8191,
        "input_cost_per_token": 0.000001,
        "output_cost_per_token": 0.000003,
        "litellm_provider": "azure_ai",
        "supports_function_calling": true,
        "mode": "chat"
    },
    "azure_ai/Meta-Llama-3-70B-Instruct": {
        "max_tokens": 8192,
        "max_input_tokens": 8192,
        "max_output_tokens": 8192,
        "input_cost_per_token": 0.0000011,
        "output_cost_per_token": 0.00000037,
        "litellm_provider": "azure_ai",
        "mode": "chat"
    },
    "azure_ai/Meta-Llama-31-8B-Instruct": {
        "max_tokens": 128000,
        "max_input_tokens": 128000,
        "max_output_tokens": 128000,
        "input_cost_per_token": 0.0000003,
        "output_cost_per_token": 0.00000061,
        "litellm_provider": "azure_ai",
        "mode": "chat",
        "source":"https://azuremarketplace.microsoft.com/en-us/marketplace/apps/metagenai.meta-llama-3-1-8b-instruct-offer?tab=PlansAndPrice"
    },
    "azure_ai/Meta-Llama-31-70B-Instruct": {
        "max_tokens": 128000,
        "max_input_tokens": 128000,
        "max_output_tokens": 128000,
        "input_cost_per_token": 0.00000268,
        "output_cost_per_token": 0.00000354,
        "litellm_provider": "azure_ai",
        "mode": "chat",
        "source":"https://azuremarketplace.microsoft.com/en-us/marketplace/apps/metagenai.meta-llama-3-1-70b-instruct-offer?tab=PlansAndPrice"
    },
    "azure_ai/Meta-Llama-31-405B-Instruct": {
        "max_tokens": 128000,
        "max_input_tokens": 128000,
        "max_output_tokens": 128000,
        "input_cost_per_token": 0.00000533,
        "output_cost_per_token": 0.000016,
        "litellm_provider": "azure_ai",
        "mode": "chat",
        "source":"https://azuremarketplace.microsoft.com/en-us/marketplace/apps/metagenai.meta-llama-3-1-405b-instruct-offer?tab=PlansAndPrice"
    },
    "babbage-002": {
        "max_tokens": 16384,
        "max_input_tokens": 16384,
        "max_output_tokens": 4096,
        "input_cost_per_token": 0.0000004,
        "output_cost_per_token": 0.0000004,
        "litellm_provider": "text-completion-openai",
        "mode": "completion"
    },
    "davinci-002": {
        "max_tokens": 16384,
        "max_input_tokens": 16384,
        "max_output_tokens": 4096,
        "input_cost_per_token": 0.000002,
        "output_cost_per_token": 0.000002,
        "litellm_provider": "text-completion-openai",
        "mode": "completion"
    },    
    "gpt-3.5-turbo-instruct": {
        "max_tokens": 4096,
        "max_input_tokens": 8192,
        "max_output_tokens": 4096,
        "input_cost_per_token": 0.0000015,
        "output_cost_per_token": 0.000002,
        "litellm_provider": "text-completion-openai",
        "mode": "completion"
    },
    "gpt-3.5-turbo-instruct-0914": {
        "max_tokens": 4097,
        "max_input_tokens": 8192,
        "max_output_tokens": 4097,
        "input_cost_per_token": 0.0000015,
        "output_cost_per_token": 0.000002,
        "litellm_provider": "text-completion-openai",
        "mode": "completion"

    },
    "claude-instant-1": {
        "max_tokens": 8191,
        "max_input_tokens": 100000,
        "max_output_tokens": 8191,
        "input_cost_per_token": 0.00000163,
        "output_cost_per_token": 0.00000551,
        "litellm_provider": "anthropic",
        "mode": "chat"
    },
    "mistral/mistral-tiny": {
        "max_tokens": 8191,
        "max_input_tokens": 32000,
        "max_output_tokens": 8191,
        "input_cost_per_token": 0.00000025,
        "output_cost_per_token": 0.00000025,
        "litellm_provider": "mistral",
        "mode": "chat",
        "supports_assistant_prefill": true
    },
    "mistral/mistral-small": {
        "max_tokens": 8191,
        "max_input_tokens": 32000,
        "max_output_tokens": 8191,
        "input_cost_per_token": 0.000001,
        "output_cost_per_token": 0.000003,
        "litellm_provider": "mistral",
        "supports_function_calling": true,
        "mode": "chat",
        "supports_assistant_prefill": true
    },
    "mistral/mistral-small-latest": {
        "max_tokens": 8191,
        "max_input_tokens": 32000,
        "max_output_tokens": 8191,
        "input_cost_per_token": 0.000001,
        "output_cost_per_token": 0.000003,
        "litellm_provider": "mistral",
        "supports_function_calling": true,
        "mode": "chat",
        "supports_assistant_prefill": true
    },
    "mistral/mistral-medium": {
        "max_tokens": 8191,
        "max_input_tokens": 32000,
        "max_output_tokens": 8191,
        "input_cost_per_token": 0.0000027,
        "output_cost_per_token": 0.0000081,
        "litellm_provider": "mistral",
        "mode": "chat",
        "supports_assistant_prefill": true
    },
    "mistral/mistral-medium-latest": {
        "max_tokens": 8191,
        "max_input_tokens": 32000,
        "max_output_tokens": 8191,
        "input_cost_per_token": 0.0000027,
        "output_cost_per_token": 0.0000081,
        "litellm_provider": "mistral",
        "mode": "chat",
        "supports_assistant_prefill": true
    },
    "mistral/mistral-medium-2312": {
        "max_tokens": 8191,
        "max_input_tokens": 32000,
        "max_output_tokens": 8191,
        "input_cost_per_token": 0.0000027,
        "output_cost_per_token": 0.0000081,
        "litellm_provider": "mistral",
        "mode": "chat",
        "supports_assistant_prefill": true
    },
    "mistral/mistral-large-latest": {
        "max_tokens": 128000,
        "max_input_tokens": 128000,
        "max_output_tokens": 128000,
        "input_cost_per_token": 0.000003,
        "output_cost_per_token": 0.000009,
        "litellm_provider": "mistral",
        "mode": "chat",
        "supports_function_calling": true,
        "supports_assistant_prefill": true
    },
    "mistral/mistral-large-2402": {
        "max_tokens": 8191,
        "max_input_tokens": 32000,
        "max_output_tokens": 8191,
        "input_cost_per_token": 0.000004,
        "output_cost_per_token": 0.000012,
        "litellm_provider": "mistral",
        "mode": "chat",
        "supports_function_calling": true,
        "supports_assistant_prefill": true
    },
    "mistral/mistral-large-2407": {
        "max_tokens": 128000,
        "max_input_tokens": 128000,
        "max_output_tokens": 128000,
        "input_cost_per_token": 0.000003,
        "output_cost_per_token": 0.000009,
        "litellm_provider": "mistral",
        "mode": "chat",
        "supports_function_calling": true,
        "supports_assistant_prefill": true
    },
    "mistral/open-mistral-7b": {
        "max_tokens": 8191,
        "max_input_tokens": 32000,
        "max_output_tokens": 8191,
        "input_cost_per_token": 0.00000025,
        "output_cost_per_token": 0.00000025,
        "litellm_provider": "mistral",
        "mode": "chat",
        "supports_assistant_prefill": true
    },
    "mistral/open-mixtral-8x7b": {
        "max_tokens": 8191,
        "max_input_tokens": 32000,
        "max_output_tokens": 8191,
        "input_cost_per_token": 0.0000007,
        "output_cost_per_token": 0.0000007,
        "litellm_provider": "mistral",
        "mode": "chat",
        "supports_function_calling": true,
        "supports_assistant_prefill": true
    },
    "mistral/open-mixtral-8x22b": {
        "max_tokens": 8191,
        "max_input_tokens": 64000,
        "max_output_tokens": 8191,
        "input_cost_per_token": 0.000002,
        "output_cost_per_token": 0.000006,
        "litellm_provider": "mistral",
        "mode": "chat",
        "supports_function_calling": true,
        "supports_assistant_prefill": true
    },
    "mistral/codestral-latest": {
        "max_tokens": 8191,
        "max_input_tokens": 32000,
        "max_output_tokens": 8191,
        "input_cost_per_token": 0.000001,
        "output_cost_per_token": 0.000003,
        "litellm_provider": "mistral",
        "mode": "chat",
        "supports_assistant_prefill": true
    },
    "mistral/codestral-2405": {
        "max_tokens": 8191,
        "max_input_tokens": 32000,
        "max_output_tokens": 8191,
        "input_cost_per_token": 0.000001,
        "output_cost_per_token": 0.000003,
        "litellm_provider": "mistral",
        "mode": "chat",
        "supports_assistant_prefill": true
    },
    "mistral/open-mistral-nemo": {
        "max_tokens": 128000,
        "max_input_tokens": 128000,
        "max_output_tokens": 128000,
        "input_cost_per_token":  0.0000003,
        "output_cost_per_token": 0.0000003,
        "litellm_provider": "mistral",
        "mode": "chat",
        "source": "https://mistral.ai/technology/",
        "supports_assistant_prefill": true
    },
    "mistral/open-mistral-nemo-2407": {
        "max_tokens": 128000,
        "max_input_tokens": 128000,
        "max_output_tokens": 128000,
        "input_cost_per_token":  0.0000003,
        "output_cost_per_token": 0.0000003,
        "litellm_provider": "mistral",
        "mode": "chat",
        "source": "https://mistral.ai/technology/",
        "supports_assistant_prefill": true
    },
    "mistral/open-codestral-mamba": {
        "max_tokens": 256000,
        "max_input_tokens": 256000,
        "max_output_tokens": 256000,
        "input_cost_per_token": 0.00000025,
        "output_cost_per_token": 0.00000025,
        "litellm_provider": "mistral",
        "mode": "chat",
        "source": "https://mistral.ai/technology/",
        "supports_assistant_prefill": true
    },
    "mistral/codestral-mamba-latest": {
        "max_tokens": 256000,
        "max_input_tokens": 256000,
        "max_output_tokens": 256000,
        "input_cost_per_token": 0.00000025,
        "output_cost_per_token": 0.00000025,
        "litellm_provider": "mistral",
        "mode": "chat",
        "source": "https://mistral.ai/technology/",
        "supports_assistant_prefill": true
    },
    "mistral/mistral-embed": {
        "max_tokens": 8192,
        "max_input_tokens": 8192,
        "input_cost_per_token": 0.0000001,
        "litellm_provider": "mistral",
        "mode": "embedding"
    },
    "deepseek-chat": {
        "max_tokens": 4096,
        "max_input_tokens": 32000,
        "max_output_tokens": 4096,
        "input_cost_per_token": 0.00000014,
        "input_cost_per_token_cache_hit": 0.000000014,
        "output_cost_per_token": 0.00000028,
        "litellm_provider": "deepseek",
        "mode": "chat",
        "supports_function_calling": true, 
        "supports_assistant_prefill": true,
        "supports_tool_choice": true
    },
    "codestral/codestral-latest": {
        "max_tokens": 8191,
        "max_input_tokens": 32000,
        "max_output_tokens": 8191,
        "input_cost_per_token": 0.000000,
        "output_cost_per_token": 0.000000,
        "litellm_provider": "codestral",
        "mode": "chat",
        "source": "https://docs.mistral.ai/capabilities/code_generation/",
        "supports_assistant_prefill": true
    },
    "codestral/codestral-2405": {
        "max_tokens": 8191,
        "max_input_tokens": 32000,
        "max_output_tokens": 8191,
        "input_cost_per_token": 0.000000,
        "output_cost_per_token": 0.000000,
        "litellm_provider": "codestral",
        "mode": "chat",
        "source": "https://docs.mistral.ai/capabilities/code_generation/",
        "supports_assistant_prefill": true
    },
    "text-completion-codestral/codestral-latest": {
        "max_tokens": 8191,
        "max_input_tokens": 32000,
        "max_output_tokens": 8191,
        "input_cost_per_token": 0.000000,
        "output_cost_per_token": 0.000000,
        "litellm_provider": "text-completion-codestral",
        "mode": "completion",
        "source": "https://docs.mistral.ai/capabilities/code_generation/"
    },
    "text-completion-codestral/codestral-2405": {
        "max_tokens": 8191,
        "max_input_tokens": 32000,
        "max_output_tokens": 8191,
        "input_cost_per_token": 0.000000,
        "output_cost_per_token": 0.000000,
        "litellm_provider": "text-completion-codestral",
        "mode": "completion",
        "source": "https://docs.mistral.ai/capabilities/code_generation/"
    },
    "deepseek-coder": {
        "max_tokens": 4096,
        "max_input_tokens": 128000,
        "max_output_tokens": 4096,
        "input_cost_per_token": 0.00000014,
        "input_cost_per_token_cache_hit": 0.000000014,
        "output_cost_per_token": 0.00000028,
        "litellm_provider": "deepseek",
        "mode": "chat",
        "supports_function_calling": true, 
        "supports_assistant_prefill": true,
        "supports_tool_choice": true
    },
    "groq/llama2-70b-4096": {
        "max_tokens": 4096,
        "max_input_tokens": 4096,
        "max_output_tokens": 4096,
        "input_cost_per_token": 0.00000070,
        "output_cost_per_token": 0.00000080,
        "litellm_provider": "groq",
        "mode": "chat",
        "supports_function_calling": true
    },
    "groq/llama3-8b-8192": {
        "max_tokens": 8192,
        "max_input_tokens": 8192,
        "max_output_tokens": 8192,
        "input_cost_per_token": 0.00000005,
        "output_cost_per_token": 0.00000008,
        "litellm_provider": "groq",
        "mode": "chat",
        "supports_function_calling": true
    },
    "groq/llama3-70b-8192": {
        "max_tokens": 8192,
        "max_input_tokens": 8192,
        "max_output_tokens": 8192,
        "input_cost_per_token": 0.00000059,
        "output_cost_per_token": 0.00000079,
        "litellm_provider": "groq",
        "mode": "chat",
        "supports_function_calling": true
    },
    "groq/llama-3.1-8b-instant": {
        "max_tokens": 8192,
        "max_input_tokens": 8192,
        "max_output_tokens": 8192,
        "input_cost_per_token": 0.00000059,
        "output_cost_per_token": 0.00000079,
        "litellm_provider": "groq",
        "mode": "chat",
        "supports_function_calling": true
    },
    "groq/llama-3.1-70b-versatile": {
        "max_tokens": 8192,
        "max_input_tokens": 8192,
        "max_output_tokens": 8192,
        "input_cost_per_token": 0.00000059,
        "output_cost_per_token": 0.00000079,
        "litellm_provider": "groq",
        "mode": "chat",
        "supports_function_calling": true
    },
    "groq/llama-3.1-405b-reasoning": {
        "max_tokens": 8192,
        "max_input_tokens": 8192,
        "max_output_tokens": 8192,
        "input_cost_per_token": 0.00000059,
        "output_cost_per_token": 0.00000079,
        "litellm_provider": "groq",
        "mode": "chat",
        "supports_function_calling": true
    },
    "groq/mixtral-8x7b-32768": {
        "max_tokens": 32768,
        "max_input_tokens": 32768,
        "max_output_tokens": 32768,
        "input_cost_per_token": 0.00000024,
        "output_cost_per_token": 0.00000024,
        "litellm_provider": "groq",
        "mode": "chat",
        "supports_function_calling": true
    },
    "groq/gemma-7b-it": {
        "max_tokens": 8192,
        "max_input_tokens": 8192,
        "max_output_tokens": 8192,
        "input_cost_per_token": 0.00000007,
        "output_cost_per_token": 0.00000007,
        "litellm_provider": "groq",
        "mode": "chat",
        "supports_function_calling": true
    },
    "groq/llama3-groq-70b-8192-tool-use-preview": {
        "max_tokens": 8192,
        "max_input_tokens": 8192,
        "max_output_tokens": 8192,
        "input_cost_per_token": 0.00000089,
        "output_cost_per_token": 0.00000089,
        "litellm_provider": "groq",
        "mode": "chat",
        "supports_function_calling": true
    },
    "groq/llama3-groq-8b-8192-tool-use-preview": {
        "max_tokens": 8192,
        "max_input_tokens": 8192,
        "max_output_tokens": 8192,
        "input_cost_per_token": 0.00000019,
        "output_cost_per_token": 0.00000019,
        "litellm_provider": "groq",
        "mode": "chat",
        "supports_function_calling": true
    },
    "cerebras/llama3.1-8b": {
        "max_tokens": 128000,
        "max_input_tokens": 128000,
        "max_output_tokens": 128000,
        "input_cost_per_token": 0.0000001,
        "output_cost_per_token": 0.0000001,
        "litellm_provider": "cerebras",
        "mode": "chat",
        "supports_function_calling": true
    },
    "cerebras/llama3.1-70b": {
        "max_tokens": 128000,
        "max_input_tokens": 128000,
        "max_output_tokens": 128000,
        "input_cost_per_token": 0.0000006,
        "output_cost_per_token": 0.0000006,
        "litellm_provider": "cerebras",
        "mode": "chat",
        "supports_function_calling": true
    },
    "friendliai/mixtral-8x7b-instruct-v0-1": {
        "max_tokens": 32768,
        "max_input_tokens": 32768,
        "max_output_tokens": 32768,
        "input_cost_per_token": 0.0000004,
        "output_cost_per_token": 0.0000004,
        "litellm_provider": "friendliai",
        "mode": "chat",
        "supports_function_calling": true
    },
    "friendliai/meta-llama-3-8b-instruct": {
        "max_tokens": 8192,
        "max_input_tokens": 8192,
        "max_output_tokens": 8192,
        "input_cost_per_token": 0.0000001,
        "output_cost_per_token": 0.0000001,
        "litellm_provider": "friendliai",
        "mode": "chat",
        "supports_function_calling": true
    },
    "friendliai/meta-llama-3-70b-instruct": {
        "max_tokens": 8192,
        "max_input_tokens": 8192,
        "max_output_tokens": 8192,
        "input_cost_per_token": 0.0000008,
        "output_cost_per_token": 0.0000008,
        "litellm_provider": "friendliai",
        "mode": "chat",
        "supports_function_calling": true
    },
    "claude-instant-1.2": {
        "max_tokens": 8191,
        "max_input_tokens": 100000,
        "max_output_tokens": 8191,
        "input_cost_per_token": 0.000000163,
        "output_cost_per_token": 0.000000551,
        "litellm_provider": "anthropic",
        "mode": "chat"
    },
    "claude-2": {
        "max_tokens": 8191,
        "max_input_tokens": 100000,
        "max_output_tokens": 8191,
        "input_cost_per_token": 0.000008,
        "output_cost_per_token": 0.000024,
        "litellm_provider": "anthropic",
        "mode": "chat"
    },
    "claude-2.1": {
        "max_tokens": 8191,
        "max_input_tokens": 200000,
        "max_output_tokens": 8191,
        "input_cost_per_token": 0.000008,
        "output_cost_per_token": 0.000024,
        "litellm_provider": "anthropic",
        "mode": "chat"
    },
    "claude-3-haiku-20240307": {
        "max_tokens": 4096,
        "max_input_tokens": 200000,
        "max_output_tokens": 4096,
        "input_cost_per_token": 0.00000025,
        "output_cost_per_token": 0.00000125,
        "cache_creation_input_token_cost": 0.0000003,
        "cache_read_input_token_cost": 0.00000003,
        "litellm_provider": "anthropic",
        "mode": "chat",
        "supports_function_calling": true,
        "supports_vision": true,
        "tool_use_system_prompt_tokens": 264,
        "supports_assistant_prefill": true
    },
    "claude-3-opus-20240229": {
        "max_tokens": 4096,
        "max_input_tokens": 200000,
        "max_output_tokens": 4096,
        "input_cost_per_token": 0.000015,
        "output_cost_per_token": 0.000075,
        "cache_creation_input_token_cost": 0.00001875,
        "cache_read_input_token_cost": 0.0000015,
        "litellm_provider": "anthropic",
        "mode": "chat",
        "supports_function_calling": true,
        "supports_vision": true,
        "tool_use_system_prompt_tokens": 395,
        "supports_assistant_prefill": true
    },
    "claude-3-sonnet-20240229": {
        "max_tokens": 4096,
        "max_input_tokens": 200000,
        "max_output_tokens": 4096,
        "input_cost_per_token": 0.000003,
        "output_cost_per_token": 0.000015,
        "litellm_provider": "anthropic",
        "mode": "chat",
        "supports_function_calling": true,
        "supports_vision": true,
        "tool_use_system_prompt_tokens": 159,
        "supports_assistant_prefill": true
    },
    "claude-3-5-sonnet-20240620": {
        "max_tokens": 4096,
        "max_input_tokens": 200000,
        "max_output_tokens": 4096,
        "input_cost_per_token": 0.000003,
        "output_cost_per_token": 0.000015,
        "cache_creation_input_token_cost": 0.00000375,
        "cache_read_input_token_cost": 0.0000003,
        "litellm_provider": "anthropic",
        "mode": "chat",
        "supports_function_calling": true,
        "supports_vision": true,
        "tool_use_system_prompt_tokens": 159,
        "supports_assistant_prefill": true
    },
    "text-bison": {
        "max_tokens": 2048,
        "max_input_tokens": 8192,
        "max_output_tokens": 2048,
        "input_cost_per_character": 0.00000025,
        "output_cost_per_character": 0.0000005,
        "litellm_provider": "vertex_ai-text-models",
        "mode": "completion",
        "source": "https://cloud.google.com/vertex-ai/generative-ai/docs/learn/models#foundation_models"
    },
    "text-bison@001": {
        "max_tokens": 1024,
        "max_input_tokens": 8192,
        "max_output_tokens": 1024,
        "input_cost_per_character": 0.00000025,
        "output_cost_per_character": 0.0000005,
        "litellm_provider": "vertex_ai-text-models",
        "mode": "completion",
        "source": "https://cloud.google.com/vertex-ai/generative-ai/docs/learn/models#foundation_models"
    },
    "text-bison@002": {
        "max_tokens": 1024,
        "max_input_tokens": 8192,
        "max_output_tokens": 1024,
        "input_cost_per_character": 0.00000025,
        "output_cost_per_character": 0.0000005,
        "litellm_provider": "vertex_ai-text-models",
        "mode": "completion",
        "source": "https://cloud.google.com/vertex-ai/generative-ai/docs/learn/models#foundation_models"
    },
    "text-bison32k": {
        "max_tokens": 1024,
        "max_input_tokens": 8192,
        "max_output_tokens": 1024,
        "input_cost_per_token": 0.000000125,
        "output_cost_per_token": 0.000000125,
        "input_cost_per_character": 0.00000025,
        "output_cost_per_character": 0.0000005,
        "litellm_provider": "vertex_ai-text-models",
        "mode": "completion",
        "source": "https://cloud.google.com/vertex-ai/generative-ai/docs/learn/models#foundation_models"
    },
    "text-bison32k@002": {
        "max_tokens": 1024,
        "max_input_tokens": 8192,
        "max_output_tokens": 1024,
        "input_cost_per_token": 0.000000125,
        "output_cost_per_token": 0.000000125,
        "input_cost_per_character": 0.00000025,
        "output_cost_per_character": 0.0000005,
        "litellm_provider": "vertex_ai-text-models",
        "mode": "completion",
        "source": "https://cloud.google.com/vertex-ai/generative-ai/docs/learn/models#foundation_models"
    },
    "text-unicorn": {
        "max_tokens": 1024,
        "max_input_tokens": 8192,
        "max_output_tokens": 1024,
        "input_cost_per_token": 0.00001,
        "output_cost_per_token": 0.000028,
        "litellm_provider": "vertex_ai-text-models",
        "mode": "completion",
        "source": "https://cloud.google.com/vertex-ai/generative-ai/docs/learn/models#foundation_models"
    },
    "text-unicorn@001": {
        "max_tokens": 1024,
        "max_input_tokens": 8192,
        "max_output_tokens": 1024,
        "input_cost_per_token": 0.00001,
        "output_cost_per_token": 0.000028,
        "litellm_provider": "vertex_ai-text-models",
        "mode": "completion",
        "source": "https://cloud.google.com/vertex-ai/generative-ai/docs/learn/models#foundation_models"
    },
    "chat-bison": {
        "max_tokens": 4096,
        "max_input_tokens": 8192,
        "max_output_tokens": 4096,
        "input_cost_per_token": 0.000000125,
        "output_cost_per_token": 0.000000125,
        "input_cost_per_character": 0.00000025,
        "output_cost_per_character": 0.0000005,
        "litellm_provider": "vertex_ai-chat-models",
        "mode": "chat",
        "source": "https://cloud.google.com/vertex-ai/generative-ai/docs/learn/models#foundation_models"
    },
    "chat-bison@001": {
        "max_tokens": 4096,
        "max_input_tokens": 8192,
        "max_output_tokens": 4096,
        "input_cost_per_token": 0.000000125,
        "output_cost_per_token": 0.000000125,
        "input_cost_per_character": 0.00000025,
        "output_cost_per_character": 0.0000005,
        "litellm_provider": "vertex_ai-chat-models",
        "mode": "chat",
        "source": "https://cloud.google.com/vertex-ai/generative-ai/docs/learn/models#foundation_models"
    },
    "chat-bison@002": {
        "max_tokens": 4096,
        "max_input_tokens": 8192,
        "max_output_tokens": 4096,
        "input_cost_per_token": 0.000000125,
        "output_cost_per_token": 0.000000125,
        "input_cost_per_character": 0.00000025,
        "output_cost_per_character": 0.0000005,
        "litellm_provider": "vertex_ai-chat-models",
        "mode": "chat",
        "source": "https://cloud.google.com/vertex-ai/generative-ai/docs/learn/models#foundation_models"
    },
    "chat-bison-32k": {
        "max_tokens": 8192,
        "max_input_tokens": 32000,
        "max_output_tokens": 8192,
        "input_cost_per_token": 0.000000125,
        "output_cost_per_token": 0.000000125,
        "input_cost_per_character": 0.00000025,
        "output_cost_per_character": 0.0000005,
        "litellm_provider": "vertex_ai-chat-models",
        "mode": "chat",
        "source": "https://cloud.google.com/vertex-ai/generative-ai/docs/learn/models#foundation_models"
    },
    "chat-bison-32k@002": {
        "max_tokens": 8192,
        "max_input_tokens": 32000,
        "max_output_tokens": 8192,
        "input_cost_per_token": 0.000000125,
        "output_cost_per_token": 0.000000125,
        "input_cost_per_character": 0.00000025,
        "output_cost_per_character": 0.0000005,
        "litellm_provider": "vertex_ai-chat-models",
        "mode": "chat",
        "source": "https://cloud.google.com/vertex-ai/generative-ai/docs/learn/models#foundation_models"
    },
    "code-bison": {
        "max_tokens": 1024,
        "max_input_tokens": 6144,
        "max_output_tokens": 1024,
        "input_cost_per_token": 0.000000125,
        "output_cost_per_token": 0.000000125,
        "input_cost_per_character": 0.00000025,
        "output_cost_per_character": 0.0000005,
        "litellm_provider": "vertex_ai-code-text-models",
        "mode": "chat",
        "source": "https://cloud.google.com/vertex-ai/generative-ai/docs/learn/models#foundation_models"
    },
    "code-bison@001": {
        "max_tokens": 1024,
        "max_input_tokens": 6144,
        "max_output_tokens": 1024,
        "input_cost_per_token": 0.000000125,
        "output_cost_per_token": 0.000000125,
        "input_cost_per_character": 0.00000025,
        "output_cost_per_character": 0.0000005,
        "litellm_provider": "vertex_ai-code-text-models",
        "mode": "completion",
        "source": "https://cloud.google.com/vertex-ai/generative-ai/docs/learn/models#foundation_models"
    },
    "code-bison@002": {
        "max_tokens": 1024,
        "max_input_tokens": 6144,
        "max_output_tokens": 1024,
        "input_cost_per_token": 0.000000125,
        "output_cost_per_token": 0.000000125,
        "input_cost_per_character": 0.00000025,
        "output_cost_per_character": 0.0000005,
        "litellm_provider": "vertex_ai-code-text-models",
        "mode": "completion",
        "source": "https://cloud.google.com/vertex-ai/generative-ai/docs/learn/models#foundation_models"
    },
    "code-bison32k": {
        "max_tokens": 1024,
        "max_input_tokens": 6144,
        "max_output_tokens": 1024,
        "input_cost_per_token": 0.000000125,
        "output_cost_per_token": 0.000000125,
        "input_cost_per_character": 0.00000025,
        "output_cost_per_character": 0.0000005,
        "litellm_provider": "vertex_ai-code-text-models",
        "mode": "completion",
        "source": "https://cloud.google.com/vertex-ai/generative-ai/docs/learn/models#foundation_models"
    },
    "code-bison-32k@002": {
        "max_tokens": 1024,
        "max_input_tokens": 6144,
        "max_output_tokens": 1024,
        "input_cost_per_token": 0.000000125,
        "output_cost_per_token": 0.000000125,
        "input_cost_per_character": 0.00000025,
        "output_cost_per_character": 0.0000005,
        "litellm_provider": "vertex_ai-code-text-models",
        "mode": "completion",
        "source": "https://cloud.google.com/vertex-ai/generative-ai/docs/learn/models#foundation_models"
    },
    "code-gecko@001": {
        "max_tokens": 64,
        "max_input_tokens": 2048,
        "max_output_tokens": 64,
        "input_cost_per_token": 0.000000125,
        "output_cost_per_token": 0.000000125,
        "litellm_provider": "vertex_ai-code-text-models",
        "mode": "completion",
        "source": "https://cloud.google.com/vertex-ai/generative-ai/docs/learn/models#foundation_models"
    },
    "code-gecko@002": {
        "max_tokens": 64,
        "max_input_tokens": 2048,
        "max_output_tokens": 64,
        "input_cost_per_token": 0.000000125,
        "output_cost_per_token": 0.000000125,
        "litellm_provider": "vertex_ai-code-text-models",
        "mode": "completion",
        "source": "https://cloud.google.com/vertex-ai/generative-ai/docs/learn/models#foundation_models"
    },
    "code-gecko": {
        "max_tokens": 64,
        "max_input_tokens": 2048,
        "max_output_tokens": 64,
        "input_cost_per_token": 0.000000125,
        "output_cost_per_token": 0.000000125,
        "litellm_provider": "vertex_ai-code-text-models",
        "mode": "completion",
        "source": "https://cloud.google.com/vertex-ai/generative-ai/docs/learn/models#foundation_models"
    },
    "code-gecko-latest": {
        "max_tokens": 64,
        "max_input_tokens": 2048,
        "max_output_tokens": 64,
        "input_cost_per_token": 0.000000125,
        "output_cost_per_token": 0.000000125,
        "litellm_provider": "vertex_ai-code-text-models",
        "mode": "completion",
        "source": "https://cloud.google.com/vertex-ai/generative-ai/docs/learn/models#foundation_models"
    },
    "codechat-bison@latest": {
        "max_tokens": 1024,
        "max_input_tokens": 6144,
        "max_output_tokens": 1024,
        "input_cost_per_token": 0.000000125,
        "output_cost_per_token": 0.000000125,
        "input_cost_per_character": 0.00000025,
        "output_cost_per_character": 0.0000005,
        "litellm_provider": "vertex_ai-code-chat-models",
        "mode": "chat",
        "source": "https://cloud.google.com/vertex-ai/generative-ai/docs/learn/models#foundation_models"
    },
    "codechat-bison": {
        "max_tokens": 1024,
        "max_input_tokens": 6144,
        "max_output_tokens": 1024,
        "input_cost_per_token": 0.000000125,
        "output_cost_per_token": 0.000000125,
        "input_cost_per_character": 0.00000025,
        "output_cost_per_character": 0.0000005,
        "litellm_provider": "vertex_ai-code-chat-models",
        "mode": "chat",
        "source": "https://cloud.google.com/vertex-ai/generative-ai/docs/learn/models#foundation_models"
    },
    "codechat-bison@001": {
        "max_tokens": 1024,
        "max_input_tokens": 6144,
        "max_output_tokens": 1024,
        "input_cost_per_token": 0.000000125,
        "output_cost_per_token": 0.000000125,
        "input_cost_per_character": 0.00000025,
        "output_cost_per_character": 0.0000005,
        "litellm_provider": "vertex_ai-code-chat-models",
        "mode": "chat",
        "source": "https://cloud.google.com/vertex-ai/generative-ai/docs/learn/models#foundation_models"
    },
    "codechat-bison@002": {
        "max_tokens": 1024,
        "max_input_tokens": 6144,
        "max_output_tokens": 1024,
        "input_cost_per_token": 0.000000125,
        "output_cost_per_token": 0.000000125,
        "input_cost_per_character": 0.00000025,
        "output_cost_per_character": 0.0000005,
        "litellm_provider": "vertex_ai-code-chat-models",
        "mode": "chat",
        "source": "https://cloud.google.com/vertex-ai/generative-ai/docs/learn/models#foundation_models"
    },
    "codechat-bison-32k": {
        "max_tokens": 8192,
        "max_input_tokens": 32000,
        "max_output_tokens": 8192,
        "input_cost_per_token": 0.000000125,
        "output_cost_per_token": 0.000000125,
        "input_cost_per_character": 0.00000025,
        "output_cost_per_character": 0.0000005,
        "litellm_provider": "vertex_ai-code-chat-models",
        "mode": "chat",
        "source": "https://cloud.google.com/vertex-ai/generative-ai/docs/learn/models#foundation_models"
    },
    "codechat-bison-32k@002": {
        "max_tokens": 8192,
        "max_input_tokens": 32000,
        "max_output_tokens": 8192,
        "input_cost_per_token": 0.000000125,
        "output_cost_per_token": 0.000000125,
        "input_cost_per_character": 0.00000025,
        "output_cost_per_character": 0.0000005,
        "litellm_provider": "vertex_ai-code-chat-models",
        "mode": "chat",
        "source": "https://cloud.google.com/vertex-ai/generative-ai/docs/learn/models#foundation_models"
    },
    "gemini-pro": {
        "max_tokens": 8192,
        "max_input_tokens": 32760,
        "max_output_tokens": 8192,
        "input_cost_per_image": 0.0025,
        "input_cost_per_video_per_second": 0.002,
        "input_cost_per_token": 0.0000005, 
        "input_cost_per_character": 0.000000125, 
        "output_cost_per_token": 0.0000015,
        "output_cost_per_character": 0.000000375,
        "litellm_provider": "vertex_ai-language-models",
        "mode": "chat",
        "supports_function_calling": true,
        "source": "https://cloud.google.com/vertex-ai/generative-ai/pricing"
    },
    "gemini-1.0-pro": { 
        "max_tokens": 8192,
        "max_input_tokens": 32760,
        "max_output_tokens": 8192,
        "input_cost_per_image": 0.0025,
        "input_cost_per_video_per_second": 0.002,
        "input_cost_per_token": 0.0000005, 
        "input_cost_per_character": 0.000000125, 
        "output_cost_per_token": 0.0000015,
        "output_cost_per_character": 0.000000375,
        "litellm_provider": "vertex_ai-language-models",
        "mode": "chat",
        "supports_function_calling": true,
        "source": "https://cloud.google.com/vertex-ai/generative-ai/pricing#google_models"
    },
    "gemini-1.0-pro-001": { 
        "max_tokens": 8192,
        "max_input_tokens": 32760,
        "max_output_tokens": 8192,
        "input_cost_per_image": 0.0025,
        "input_cost_per_video_per_second": 0.002,
        "input_cost_per_token": 0.0000005, 
        "input_cost_per_character": 0.000000125, 
        "output_cost_per_token": 0.0000015,
        "output_cost_per_character": 0.000000375,
        "litellm_provider": "vertex_ai-language-models",
        "mode": "chat",
        "supports_function_calling": true,
        "source": "https://cloud.google.com/vertex-ai/generative-ai/docs/learn/models#foundation_models"
    },
    "gemini-1.0-ultra": {
        "max_tokens": 8192,
        "max_input_tokens": 8192,
        "max_output_tokens": 2048,
        "input_cost_per_image": 0.0025,
        "input_cost_per_video_per_second": 0.002,
        "input_cost_per_token": 0.0000005, 
        "input_cost_per_character": 0.000000125, 
        "output_cost_per_token": 0.0000015,
        "output_cost_per_character": 0.000000375,
        "litellm_provider": "vertex_ai-language-models",
        "mode": "chat",
        "supports_function_calling": true,
        "source": "As of Jun, 2024. There is no available doc on vertex ai pricing gemini-1.0-ultra-001. Using gemini-1.0-pro pricing. Got max_tokens info here: https://cloud.google.com/vertex-ai/generative-ai/docs/learn/models#foundation_models"
    },
    "gemini-1.0-ultra-001": {
        "max_tokens": 8192,
        "max_input_tokens": 8192,
        "max_output_tokens": 2048,
        "input_cost_per_image": 0.0025,
        "input_cost_per_video_per_second": 0.002,
        "input_cost_per_token": 0.0000005, 
        "input_cost_per_character": 0.000000125, 
        "output_cost_per_token": 0.0000015,
        "output_cost_per_character": 0.000000375,
        "litellm_provider": "vertex_ai-language-models",
        "mode": "chat",
        "supports_function_calling": true,
        "source": "As of Jun, 2024. There is no available doc on vertex ai pricing gemini-1.0-ultra-001. Using gemini-1.0-pro pricing. Got max_tokens info here: https://cloud.google.com/vertex-ai/generative-ai/docs/learn/models#foundation_models"
    },
    "gemini-1.0-pro-002": { 
        "max_tokens": 8192,
        "max_input_tokens": 32760,
        "max_output_tokens": 8192,
        "input_cost_per_image": 0.0025,
        "input_cost_per_video_per_second": 0.002,
        "input_cost_per_token": 0.0000005, 
        "input_cost_per_character": 0.000000125, 
        "output_cost_per_token": 0.0000015,
        "output_cost_per_character": 0.000000375,
        "litellm_provider": "vertex_ai-language-models",
        "mode": "chat",
        "supports_function_calling": true,
        "source": "https://cloud.google.com/vertex-ai/generative-ai/docs/learn/models#foundation_models"
    },
    "gemini-1.5-pro": { 
        "max_tokens": 8192,
        "max_input_tokens": 2097152,
        "max_output_tokens": 8192,
        "input_cost_per_image": 0.001315,
        "input_cost_per_audio_per_second": 0.000125,
        "input_cost_per_video_per_second": 0.001315,
        "input_cost_per_token": 0.000005, 
        "input_cost_per_character": 0.00000125, 
        "input_cost_per_token_above_128k_tokens": 0.00001, 
        "input_cost_per_character_above_128k_tokens": 0.0000025, 
        "output_cost_per_token": 0.000015,
        "output_cost_per_character": 0.00000375,
        "output_cost_per_token_above_128k_tokens": 0.00003,
        "output_cost_per_character_above_128k_tokens": 0.0000075,
        "output_cost_per_image": 0.00263,
        "output_cost_per_video_per_second": 0.00263,
        "output_cost_per_audio_per_second": 0.00025,
        "litellm_provider": "vertex_ai-language-models",
        "mode": "chat",
        "supports_system_messages": true,
        "supports_function_calling": true,
        "supports_tool_choice": true, 
        "supports_response_schema": true, 
        "source": "https://cloud.google.com/vertex-ai/generative-ai/docs/learn/models#foundation_models"
    },
    "gemini-1.5-pro-001": { 
        "max_tokens": 8192,
        "max_input_tokens": 1000000,
        "max_output_tokens": 8192,
        "input_cost_per_image": 0.001315,
        "input_cost_per_audio_per_second": 0.000125,
        "input_cost_per_video_per_second": 0.001315,
        "input_cost_per_token": 0.000005, 
        "input_cost_per_character": 0.00000125, 
        "input_cost_per_token_above_128k_tokens": 0.00001, 
        "input_cost_per_character_above_128k_tokens": 0.0000025, 
        "output_cost_per_token": 0.000015,
        "output_cost_per_character": 0.00000375,
        "output_cost_per_token_above_128k_tokens": 0.00003,
        "output_cost_per_character_above_128k_tokens": 0.0000075,
        "output_cost_per_image": 0.00263,
        "output_cost_per_video_per_second": 0.00263,
        "output_cost_per_audio_per_second": 0.00025,
        "litellm_provider": "vertex_ai-language-models",
        "mode": "chat",
        "supports_system_messages": true,
        "supports_function_calling": true,
        "supports_tool_choice": true, 
        "supports_response_schema": true, 
        "source": "https://cloud.google.com/vertex-ai/generative-ai/docs/learn/models#foundation_models"
    },
    "gemini-1.5-pro-preview-0514": { 
        "max_tokens": 8192,
        "max_input_tokens": 1000000,
        "max_output_tokens": 8192,
        "input_cost_per_image": 0.001315,
        "input_cost_per_audio_per_second": 0.000125,
        "input_cost_per_video_per_second": 0.001315,
        "input_cost_per_token": 0.000005, 
        "input_cost_per_character": 0.00000125, 
        "input_cost_per_token_above_128k_tokens": 0.00001, 
        "input_cost_per_character_above_128k_tokens": 0.0000025, 
        "output_cost_per_token": 0.000015,
        "output_cost_per_character": 0.00000375,
        "output_cost_per_token_above_128k_tokens": 0.00003,
        "output_cost_per_character_above_128k_tokens": 0.0000075,
        "output_cost_per_image": 0.00263,
        "output_cost_per_video_per_second": 0.00263,
        "output_cost_per_audio_per_second": 0.00025,
        "litellm_provider": "vertex_ai-language-models",
        "mode": "chat",
        "supports_system_messages": true,
        "supports_function_calling": true,
        "supports_tool_choice": true, 
        "supports_response_schema": true, 
        "source": "https://cloud.google.com/vertex-ai/generative-ai/docs/learn/models#foundation_models"
    },
    "gemini-1.5-pro-preview-0215": { 
        "max_tokens": 8192,
        "max_input_tokens": 1000000,
        "max_output_tokens": 8192,
        "input_cost_per_image": 0.001315,
        "input_cost_per_audio_per_second": 0.000125,
        "input_cost_per_video_per_second": 0.001315,
        "input_cost_per_token": 0.000005, 
        "input_cost_per_character": 0.00000125, 
        "input_cost_per_token_above_128k_tokens": 0.00001, 
        "input_cost_per_character_above_128k_tokens": 0.0000025, 
        "output_cost_per_token": 0.000015,
        "output_cost_per_character": 0.00000375,
        "output_cost_per_token_above_128k_tokens": 0.00003,
        "output_cost_per_character_above_128k_tokens": 0.0000075,
        "output_cost_per_image": 0.00263,
        "output_cost_per_video_per_second": 0.00263,
        "output_cost_per_audio_per_second": 0.00025,
        "litellm_provider": "vertex_ai-language-models",
        "mode": "chat",
        "supports_system_messages": true,
        "supports_function_calling": true,
        "supports_tool_choice": true, 
        "supports_response_schema": true, 
        "source": "https://cloud.google.com/vertex-ai/generative-ai/docs/learn/models#foundation_models"
    },
    "gemini-1.5-pro-preview-0409": {
        "max_tokens": 8192,
        "max_input_tokens": 1000000,
        "max_output_tokens": 8192,
        "input_cost_per_image": 0.001315,
        "input_cost_per_audio_per_second": 0.000125,
        "input_cost_per_video_per_second": 0.001315,
        "input_cost_per_token": 0.000005, 
        "input_cost_per_character": 0.00000125, 
        "input_cost_per_token_above_128k_tokens": 0.00001, 
        "input_cost_per_character_above_128k_tokens": 0.0000025, 
        "output_cost_per_token": 0.000015,
        "output_cost_per_character": 0.00000375,
        "output_cost_per_token_above_128k_tokens": 0.00003,
        "output_cost_per_character_above_128k_tokens": 0.0000075,
        "output_cost_per_image": 0.00263,
        "output_cost_per_video_per_second": 0.00263,
        "output_cost_per_audio_per_second": 0.00025,
        "litellm_provider": "vertex_ai-language-models",
        "mode": "chat",
        "supports_function_calling": true,
        "supports_tool_choice": true,
        "supports_response_schema": true, 
        "source": "https://cloud.google.com/vertex-ai/generative-ai/docs/learn/models#foundation_models"
    },
    "gemini-1.5-flash": {
        "max_tokens": 8192,
        "max_input_tokens": 1000000,
        "max_output_tokens": 8192,
        "max_images_per_prompt": 3000,
        "max_videos_per_prompt": 10,
        "max_video_length": 1,
        "max_audio_length_hours": 8.4,
        "max_audio_per_prompt": 1,
        "max_pdf_size_mb": 30,
        "input_cost_per_image": 0.0001315,
        "input_cost_per_video_per_second": 0.0001315,
        "input_cost_per_audio_per_second": 0.000125,
        "input_cost_per_token": 0.0000005, 
        "input_cost_per_character": 0.000000125, 
        "input_cost_per_token_above_128k_tokens": 0.000001, 
        "input_cost_per_character_above_128k_tokens": 0.00000025, 
        "output_cost_per_token": 0.0000015,
        "output_cost_per_character": 0.000000375,
        "output_cost_per_token_above_128k_tokens": 0.000003,
        "output_cost_per_character_above_128k_tokens": 0.00000075,
        "output_cost_per_image": 0.000263,
        "output_cost_per_video_per_second": 0.000263,
        "output_cost_per_audio_per_second": 0.00025,
        "litellm_provider": "vertex_ai-language-models",
        "mode": "chat",
        "supports_system_messages": true,
        "supports_function_calling": true,
        "supports_vision": true,
        "source": "https://cloud.google.com/vertex-ai/generative-ai/docs/learn/models#foundation_models"
    },
    "gemini-1.5-flash-001": {
        "max_tokens": 8192,
        "max_input_tokens": 1000000,
        "max_output_tokens": 8192,
        "max_images_per_prompt": 3000,
        "max_videos_per_prompt": 10,
        "max_video_length": 1,
        "max_audio_length_hours": 8.4,
        "max_audio_per_prompt": 1,
        "max_pdf_size_mb": 30,
        "input_cost_per_image": 0.0001315,
        "input_cost_per_video_per_second": 0.0001315,
        "input_cost_per_audio_per_second": 0.000125,
        "input_cost_per_token": 0.0000005, 
        "input_cost_per_character": 0.000000125, 
        "input_cost_per_token_above_128k_tokens": 0.000001, 
        "input_cost_per_character_above_128k_tokens": 0.00000025, 
        "output_cost_per_token": 0.0000015,
        "output_cost_per_character": 0.000000375,
        "output_cost_per_token_above_128k_tokens": 0.000003,
        "output_cost_per_character_above_128k_tokens": 0.00000075,
        "output_cost_per_image": 0.000263,
        "output_cost_per_video_per_second": 0.000263,
        "output_cost_per_audio_per_second": 0.00025,
        "litellm_provider": "vertex_ai-language-models",
        "mode": "chat",
        "supports_system_messages": true,
        "supports_function_calling": true,
        "supports_vision": true,
        "source": "https://cloud.google.com/vertex-ai/generative-ai/docs/learn/models#foundation_models"
    },
    "gemini-1.5-flash-preview-0514": {
        "max_tokens": 8192,
        "max_input_tokens": 1000000,
        "max_output_tokens": 8192,
        "max_images_per_prompt": 3000,
        "max_videos_per_prompt": 10,
        "max_video_length": 1,
        "max_audio_length_hours": 8.4,
        "max_audio_per_prompt": 1,
        "max_pdf_size_mb": 30,
        "input_cost_per_image": 0.0001315,
        "input_cost_per_video_per_second": 0.0001315,
        "input_cost_per_audio_per_second": 0.000125,
        "input_cost_per_token": 0.0000005, 
        "input_cost_per_character": 0.000000125, 
        "input_cost_per_token_above_128k_tokens": 0.000001, 
        "input_cost_per_character_above_128k_tokens": 0.00000025, 
        "output_cost_per_token": 0.0000015,
        "output_cost_per_character": 0.000000375,
        "output_cost_per_token_above_128k_tokens": 0.000003,
        "output_cost_per_character_above_128k_tokens": 0.00000075,
        "output_cost_per_image": 0.000263,
        "output_cost_per_video_per_second": 0.000263,
        "output_cost_per_audio_per_second": 0.00025,
        "litellm_provider": "vertex_ai-language-models",
        "mode": "chat",
        "supports_system_messages": true,
        "supports_function_calling": true,
        "supports_vision": true,
        "source": "https://cloud.google.com/vertex-ai/generative-ai/docs/learn/models#foundation_models"
    },
    "gemini-pro-experimental": {
        "max_tokens": 8192,
        "max_input_tokens": 1000000,
        "max_output_tokens": 8192,
        "input_cost_per_token": 0,
        "output_cost_per_token": 0,
        "input_cost_per_character": 0,
        "output_cost_per_character": 0,
        "litellm_provider": "vertex_ai-language-models",
        "mode": "chat",
        "supports_function_calling": false,
        "supports_tool_choice": true, 
        "source": "https://cloud.google.com/vertex-ai/generative-ai/docs/multimodal/gemini-experimental"
    },
    "gemini-pro-flash": {
        "max_tokens": 8192,
        "max_input_tokens": 1000000,
        "max_output_tokens": 8192,
        "input_cost_per_token": 0,
        "output_cost_per_token": 0,
        "input_cost_per_character": 0,
        "output_cost_per_character": 0,
        "litellm_provider": "vertex_ai-language-models",
        "mode": "chat",
        "supports_function_calling": false,
        "supports_tool_choice": true, 
        "source": "https://cloud.google.com/vertex-ai/generative-ai/docs/multimodal/gemini-experimental"
    },
    "gemini-pro-vision": {
        "max_tokens": 2048,
        "max_input_tokens": 16384,
        "max_output_tokens": 2048,
        "max_images_per_prompt": 16,
        "max_videos_per_prompt": 1,
        "max_video_length": 2,
        "input_cost_per_token": 0.00000025, 
        "output_cost_per_token": 0.0000005,
        "litellm_provider": "vertex_ai-vision-models",
        "mode": "chat",
        "supports_function_calling": true,
        "supports_vision": true,
        "source": "https://cloud.google.com/vertex-ai/generative-ai/docs/learn/models#foundation_models"
    },
    "gemini-1.0-pro-vision": {
        "max_tokens": 2048,
        "max_input_tokens": 16384,
        "max_output_tokens": 2048,
        "max_images_per_prompt": 16,
        "max_videos_per_prompt": 1,
        "max_video_length": 2,
        "input_cost_per_token": 0.00000025, 
        "output_cost_per_token": 0.0000005,
        "litellm_provider": "vertex_ai-vision-models",
        "mode": "chat",
        "supports_function_calling": true,
        "supports_vision": true,
        "source": "https://cloud.google.com/vertex-ai/generative-ai/docs/learn/models#foundation_models"
    },
    "gemini-1.0-pro-vision-001": {
        "max_tokens": 2048,
        "max_input_tokens": 16384,
        "max_output_tokens": 2048,
        "max_images_per_prompt": 16,
        "max_videos_per_prompt": 1,
        "max_video_length": 2,
        "input_cost_per_token": 0.00000025, 
        "output_cost_per_token": 0.0000005,
        "litellm_provider": "vertex_ai-vision-models",
        "mode": "chat",
        "supports_function_calling": true,
        "supports_vision": true,
        "source": "https://cloud.google.com/vertex-ai/generative-ai/docs/learn/models#foundation_models"
    },
    "medlm-medium": {
        "max_tokens": 8192,
        "max_input_tokens": 32768,
        "max_output_tokens": 8192,
        "input_cost_per_character": 0.0000005,
        "output_cost_per_character": 0.000001,
        "litellm_provider": "vertex_ai-language-models",
        "mode": "chat",
        "source": "https://cloud.google.com/vertex-ai/generative-ai/docs/learn/models#foundation_models"
    },
    "medlm-large": {
        "max_tokens": 1024,
        "max_input_tokens": 8192,
        "max_output_tokens": 1024,
        "input_cost_per_character": 0.000005,
        "output_cost_per_character": 0.000015,
        "litellm_provider": "vertex_ai-language-models",
        "mode": "chat",
        "source": "https://cloud.google.com/vertex-ai/generative-ai/docs/learn/models#foundation_models"
    },
    "vertex_ai/claude-3-sonnet@20240229": {
        "max_tokens": 4096,
        "max_input_tokens": 200000,
        "max_output_tokens": 4096,
        "input_cost_per_token": 0.000003,
        "output_cost_per_token": 0.000015,
        "litellm_provider": "vertex_ai-anthropic_models",
        "mode": "chat",
        "supports_function_calling": true,
        "supports_vision": true,
        "supports_assistant_prefill": true
    },
    "vertex_ai/claude-3-5-sonnet@20240620": {
        "max_tokens": 4096,
        "max_input_tokens": 200000,
        "max_output_tokens": 4096,
        "input_cost_per_token": 0.000003,
        "output_cost_per_token": 0.000015,
        "litellm_provider": "vertex_ai-anthropic_models",
        "mode": "chat",
        "supports_function_calling": true,
        "supports_vision": true,
        "supports_assistant_prefill": true
    },
    "vertex_ai/claude-3-haiku@20240307": {
        "max_tokens": 4096, 
        "max_input_tokens": 200000,
        "max_output_tokens": 4096,
        "input_cost_per_token": 0.00000025,
        "output_cost_per_token": 0.00000125,
        "litellm_provider": "vertex_ai-anthropic_models",
        "mode": "chat",
        "supports_function_calling": true,
        "supports_vision": true,
        "supports_assistant_prefill": true
    },
    "vertex_ai/claude-3-opus@20240229": {
        "max_tokens": 4096,
        "max_input_tokens": 200000,
        "max_output_tokens": 4096,
        "input_cost_per_token": 0.000015,
        "output_cost_per_token": 0.000075,
        "litellm_provider": "vertex_ai-anthropic_models",
        "mode": "chat",
        "supports_function_calling": true,
        "supports_vision": true,
        "supports_assistant_prefill": true
    },
    "vertex_ai/meta/llama3-405b-instruct-maas": {
        "max_tokens": 32000,
        "max_input_tokens": 32000,
        "max_output_tokens": 32000,
        "input_cost_per_token": 0.0,
        "output_cost_per_token": 0.0,
        "litellm_provider": "vertex_ai-llama_models",
        "mode": "chat",
        "source": "https://cloud.google.com/vertex-ai/generative-ai/pricing#partner-models"
    },
    "vertex_ai/mistral-large@latest": {
        "max_tokens": 8191,
        "max_input_tokens": 128000,
        "max_output_tokens": 8191,
        "input_cost_per_token": 0.000003,
        "output_cost_per_token": 0.000009,
        "litellm_provider": "vertex_ai-mistral_models",
        "mode": "chat",
        "supports_function_calling": true
    },
    "vertex_ai/mistral-large@2407": {
        "max_tokens": 8191,
        "max_input_tokens": 128000,
        "max_output_tokens": 8191,
        "input_cost_per_token": 0.000003,
        "output_cost_per_token": 0.000009,
        "litellm_provider": "vertex_ai-mistral_models",
        "mode": "chat",
        "supports_function_calling": true
    },
    "vertex_ai/mistral-nemo@latest": {
        "max_tokens": 128000,
        "max_input_tokens": 128000,
        "max_output_tokens": 128000,
        "input_cost_per_token": 0.000003,
        "output_cost_per_token": 0.000003,
        "litellm_provider": "vertex_ai-mistral_models",
        "mode": "chat",
        "supports_function_calling": true
    },
    "vertex_ai/jamba-1.5-mini@001": {
        "max_tokens": 256000,
        "max_input_tokens": 256000,
        "max_output_tokens": 256000,
        "input_cost_per_token": 0.0000002,
        "output_cost_per_token": 0.0000004,
        "litellm_provider": "vertex_ai-ai21_models",
        "mode": "chat"
    },
    "vertex_ai/jamba-1.5-large@001": {
        "max_tokens": 256000,
        "max_input_tokens": 256000,
        "max_output_tokens": 256000,
        "input_cost_per_token": 0.000002,
        "output_cost_per_token": 0.000008,
        "litellm_provider": "vertex_ai-ai21_models",
        "mode": "chat"
    },
    "vertex_ai/jamba-1.5": {
        "max_tokens": 256000,
        "max_input_tokens": 256000,
        "max_output_tokens": 256000,
        "input_cost_per_token": 0.0000002,
        "output_cost_per_token": 0.0000004,
        "litellm_provider": "vertex_ai-ai21_models",
        "mode": "chat"
    },
    "vertex_ai/jamba-1.5-mini": {
        "max_tokens": 256000,
        "max_input_tokens": 256000,
        "max_output_tokens": 256000,
        "input_cost_per_token": 0.0000002,
        "output_cost_per_token": 0.0000004,
        "litellm_provider": "vertex_ai-ai21_models",
        "mode": "chat"
    },
    "vertex_ai/jamba-1.5-large": {
        "max_tokens": 256000,
        "max_input_tokens": 256000,
        "max_output_tokens": 256000,
        "input_cost_per_token": 0.000002,
        "output_cost_per_token": 0.000008,
        "litellm_provider": "vertex_ai-ai21_models",
        "mode": "chat"
    },
    "vertex_ai/mistral-nemo@2407": {
        "max_tokens": 128000,
        "max_input_tokens": 128000,
        "max_output_tokens": 128000,
        "input_cost_per_token": 0.000003,
        "output_cost_per_token": 0.000003,
        "litellm_provider": "vertex_ai-mistral_models",
        "mode": "chat",
        "supports_function_calling": true
    },
    "vertex_ai/codestral@latest": {
        "max_tokens": 128000,
        "max_input_tokens": 128000,
        "max_output_tokens": 128000,
        "input_cost_per_token": 0.000001,
        "output_cost_per_token": 0.000003,
        "litellm_provider": "vertex_ai-mistral_models",
        "mode": "chat",
        "supports_function_calling": true
    },
    "vertex_ai/codestral@2405": {
        "max_tokens": 128000,
        "max_input_tokens": 128000,
        "max_output_tokens": 128000,
        "input_cost_per_token": 0.000001,
        "output_cost_per_token": 0.000003,
        "litellm_provider": "vertex_ai-mistral_models",
        "mode": "chat",
        "supports_function_calling": true
    },
    "vertex_ai/imagegeneration@006": {
        "cost_per_image": 0.020,
        "litellm_provider": "vertex_ai-image-models",
        "mode": "image_generation",
        "source": "https://cloud.google.com/vertex-ai/generative-ai/pricing"
    },
    "vertex_ai/imagen-3.0-generate-001": {
        "cost_per_image": 0.04,
        "litellm_provider": "vertex_ai-image-models",
        "mode": "image_generation",
        "source": "https://cloud.google.com/vertex-ai/generative-ai/pricing"
    },
    "vertex_ai/imagen-3.0-fast-generate-001": {
        "cost_per_image": 0.02,
        "litellm_provider": "vertex_ai-image-models",
        "mode": "image_generation",
        "source": "https://cloud.google.com/vertex-ai/generative-ai/pricing"
    },
    "text-embedding-004": {
        "max_tokens": 3072,
        "max_input_tokens": 3072,
        "output_vector_size": 768,
        "input_cost_per_token": 0.00000000625,
        "output_cost_per_token": 0,
        "litellm_provider": "vertex_ai-embedding-models",
        "mode": "embedding",
        "source": "https://cloud.google.com/vertex-ai/generative-ai/docs/learn/models"
    },
    "text-multilingual-embedding-002": {
        "max_tokens": 2048,
        "max_input_tokens": 2048,
        "output_vector_size": 768,
        "input_cost_per_token": 0.00000000625,
        "output_cost_per_token": 0,
        "litellm_provider": "vertex_ai-embedding-models",
        "mode": "embedding",
        "source": "https://cloud.google.com/vertex-ai/generative-ai/docs/learn/models"
    },
    "textembedding-gecko": {
        "max_tokens": 3072,
        "max_input_tokens": 3072,
        "output_vector_size": 768,
        "input_cost_per_token": 0.00000000625,
        "output_cost_per_token": 0,
        "litellm_provider": "vertex_ai-embedding-models",
        "mode": "embedding",
        "source": "https://cloud.google.com/vertex-ai/generative-ai/docs/learn/models#foundation_models"
    },
    "textembedding-gecko-multilingual": {
        "max_tokens": 3072,
        "max_input_tokens": 3072,
        "output_vector_size": 768,
        "input_cost_per_token": 0.00000000625,
        "output_cost_per_token": 0,
        "litellm_provider": "vertex_ai-embedding-models",
        "mode": "embedding",
        "source": "https://cloud.google.com/vertex-ai/generative-ai/docs/learn/models#foundation_models"
    },
    "textembedding-gecko-multilingual@001": {
        "max_tokens": 3072,
        "max_input_tokens": 3072,
        "output_vector_size": 768,
        "input_cost_per_token": 0.00000000625,
        "output_cost_per_token": 0,
        "litellm_provider": "vertex_ai-embedding-models",
        "mode": "embedding",
        "source": "https://cloud.google.com/vertex-ai/generative-ai/docs/learn/models#foundation_models"
    },
    "textembedding-gecko@001": {
        "max_tokens": 3072,
        "max_input_tokens": 3072,
        "output_vector_size": 768,
        "input_cost_per_token": 0.00000000625,
        "output_cost_per_token": 0,
        "litellm_provider": "vertex_ai-embedding-models",
        "mode": "embedding",
        "source": "https://cloud.google.com/vertex-ai/generative-ai/docs/learn/models#foundation_models"
    },
    "textembedding-gecko@003": {
        "max_tokens": 3072,
        "max_input_tokens": 3072,
        "output_vector_size": 768,
        "input_cost_per_token": 0.00000000625,
        "output_cost_per_token": 0,
        "litellm_provider": "vertex_ai-embedding-models",
        "mode": "embedding",
        "source": "https://cloud.google.com/vertex-ai/generative-ai/docs/learn/models#foundation_models"
    },
    "text-embedding-preview-0409": {
        "max_tokens": 3072,
        "max_input_tokens": 3072,
        "output_vector_size": 768,
        "input_cost_per_token": 0.00000000625,
        "input_cost_per_token_batch_requests": 0.000000005,
        "output_cost_per_token": 0,
        "litellm_provider": "vertex_ai-embedding-models",
        "mode": "embedding",
        "source": "https://cloud.google.com/vertex-ai/generative-ai/pricing"
    },
    "text-multilingual-embedding-preview-0409":{
        "max_tokens": 3072,
        "max_input_tokens": 3072,
        "output_vector_size": 768,
        "input_cost_per_token": 0.00000000625,
        "output_cost_per_token": 0,
        "litellm_provider": "vertex_ai-embedding-models",
        "mode": "embedding",
        "source": "https://cloud.google.com/vertex-ai/generative-ai/docs/learn/models#foundation_models"
    },
    "palm/chat-bison": {
        "max_tokens": 4096,
        "max_input_tokens": 8192,
        "max_output_tokens": 4096,
        "input_cost_per_token": 0.000000125,
        "output_cost_per_token": 0.000000125,
        "litellm_provider": "palm",
        "mode": "chat",
        "source": "https://cloud.google.com/vertex-ai/generative-ai/docs/learn/models#foundation_models"
    },
    "palm/chat-bison-001": {
        "max_tokens": 4096,
        "max_input_tokens": 8192,
        "max_output_tokens": 4096,
        "input_cost_per_token": 0.000000125,
        "output_cost_per_token": 0.000000125,
        "litellm_provider": "palm",
        "mode": "chat",
        "source": "https://cloud.google.com/vertex-ai/generative-ai/docs/learn/models#foundation_models"
    },
    "palm/text-bison": {
        "max_tokens": 1024,
        "max_input_tokens": 8192,
        "max_output_tokens": 1024,
        "input_cost_per_token": 0.000000125,
        "output_cost_per_token": 0.000000125,
        "litellm_provider": "palm",
        "mode": "completion",
        "source": "https://cloud.google.com/vertex-ai/generative-ai/docs/learn/models#foundation_models"
    },
    "palm/text-bison-001": {
        "max_tokens": 1024,
        "max_input_tokens": 8192,
        "max_output_tokens": 1024,
        "input_cost_per_token": 0.000000125,
        "output_cost_per_token": 0.000000125,
        "litellm_provider": "palm",
        "mode": "completion",
        "source": "https://cloud.google.com/vertex-ai/generative-ai/docs/learn/models#foundation_models"
    },
    "palm/text-bison-safety-off": {
        "max_tokens": 1024,
        "max_input_tokens": 8192,
        "max_output_tokens": 1024,
        "input_cost_per_token": 0.000000125,
        "output_cost_per_token": 0.000000125,
        "litellm_provider": "palm",
        "mode": "completion",
        "source": "https://cloud.google.com/vertex-ai/generative-ai/docs/learn/models#foundation_models"
    },
    "palm/text-bison-safety-recitation-off": {
        "max_tokens": 1024,
        "max_input_tokens": 8192,
        "max_output_tokens": 1024,
        "input_cost_per_token": 0.000000125,
        "output_cost_per_token": 0.000000125,
        "litellm_provider": "palm",
        "mode": "completion",
        "source": "https://cloud.google.com/vertex-ai/generative-ai/docs/learn/models#foundation_models"
    },
    "gemini/gemini-1.5-flash": {
        "max_tokens": 8192,
        "max_input_tokens": 1000000,
        "max_output_tokens": 8192,
        "max_images_per_prompt": 3000,
        "max_videos_per_prompt": 10,
        "max_video_length": 1,
        "max_audio_length_hours": 8.4,
        "max_audio_per_prompt": 1,
        "max_pdf_size_mb": 30, 
        "input_cost_per_token": 0.00000035, 
        "input_cost_per_token_above_128k_tokens": 0.0000007, 
        "output_cost_per_token": 0.00000105, 
        "output_cost_per_token_above_128k_tokens": 0.0000021, 
        "litellm_provider": "gemini",
        "mode": "chat",
        "supports_system_messages": true,
        "supports_function_calling": true,
        "supports_vision": true,
        "source": "https://cloud.google.com/vertex-ai/generative-ai/docs/learn/models#foundation_models"
    },
    "gemini/gemini-1.5-flash-latest": {
        "max_tokens": 8192,
        "max_input_tokens": 1000000,
        "max_output_tokens": 8192,
        "max_images_per_prompt": 3000,
        "max_videos_per_prompt": 10,
        "max_video_length": 1,
        "max_audio_length_hours": 8.4,
        "max_audio_per_prompt": 1,
        "max_pdf_size_mb": 30, 
        "input_cost_per_token": 0.00000035, 
        "input_cost_per_token_above_128k_tokens": 0.0000007, 
        "output_cost_per_token": 0.00000105, 
        "output_cost_per_token_above_128k_tokens": 0.0000021, 
        "litellm_provider": "gemini",
        "mode": "chat",
        "supports_system_messages": true,
        "supports_function_calling": true,
        "supports_vision": true,
        "source": "https://cloud.google.com/vertex-ai/generative-ai/docs/learn/models#foundation_models"
    },
    "gemini/gemini-pro": {
        "max_tokens": 8192,
        "max_input_tokens": 32760,
        "max_output_tokens": 8192,
        "input_cost_per_token": 0.00000035, 
        "input_cost_per_token_above_128k_tokens": 0.0000007, 
        "output_cost_per_token": 0.00000105, 
        "output_cost_per_token_above_128k_tokens": 0.0000021, 
        "litellm_provider": "gemini",
        "mode": "chat",
        "supports_function_calling": true,
        "source": "https://cloud.google.com/vertex-ai/generative-ai/docs/learn/models#foundation_models"
    },
    "gemini/gemini-1.5-pro": {
        "max_tokens": 8192,
        "max_input_tokens": 2097152,
        "max_output_tokens": 8192,
        "input_cost_per_token": 0.0000035, 
        "input_cost_per_token_above_128k_tokens": 0.000007, 
        "output_cost_per_token": 0.0000105, 
        "output_cost_per_token_above_128k_tokens": 0.000021, 
        "litellm_provider": "gemini",
        "mode": "chat",
        "supports_system_messages": true,
        "supports_function_calling": true,
        "supports_vision": true,
        "supports_tool_choice": true, 
        "supports_response_schema": true, 
        "source": "https://ai.google.dev/pricing"
    },
    "gemini/gemini-1.5-pro-exp-0801": {
        "max_tokens": 8192,
        "max_input_tokens": 2097152,
        "max_output_tokens": 8192,
        "input_cost_per_token": 0.0000035,
        "input_cost_per_token_above_128k_tokens": 0.000007,
        "output_cost_per_token": 0.0000105,
        "output_cost_per_token_above_128k_tokens": 0.000021,
        "litellm_provider": "gemini",
        "mode": "chat",
        "supports_system_messages": true,
        "supports_function_calling": true,
        "supports_vision": true,
        "supports_tool_choice": true,
        "supports_response_schema": true,
        "source": "https://ai.google.dev/pricing"
    },
    "gemini/gemini-1.5-pro-exp-0827": {
        "max_tokens": 8192,
        "max_input_tokens": 2097152,
        "max_output_tokens": 8192,
        "input_cost_per_token": 0.0000035,
        "input_cost_per_token_above_128k_tokens": 0.000007,
        "output_cost_per_token": 0.0000105,
        "output_cost_per_token_above_128k_tokens": 0.000021,
        "litellm_provider": "gemini",
        "mode": "chat",
        "supports_system_messages": true,
        "supports_function_calling": true,
        "supports_vision": true,
        "supports_tool_choice": true,
        "supports_response_schema": true,
        "source": "https://ai.google.dev/pricing"
    },
    "gemini/gemini-1.5-pro-latest": {
        "max_tokens": 8192,
        "max_input_tokens": 1048576,
        "max_output_tokens": 8192,
        "input_cost_per_token": 0.0000035, 
        "input_cost_per_token_above_128k_tokens": 0.000007, 
        "output_cost_per_token": 0.00000105, 
        "output_cost_per_token_above_128k_tokens": 0.000021, 
        "litellm_provider": "gemini",
        "mode": "chat",
        "supports_system_messages": true,
        "supports_function_calling": true,
        "supports_vision": true,
        "supports_tool_choice": true, 
        "supports_response_schema": true, 
        "source": "https://ai.google.dev/pricing"
    },
    "gemini/gemini-pro-vision": {
        "max_tokens": 2048,
        "max_input_tokens": 30720,
        "max_output_tokens": 2048,
        "input_cost_per_token": 0.00000035, 
        "input_cost_per_token_above_128k_tokens": 0.0000007, 
        "output_cost_per_token": 0.00000105, 
        "output_cost_per_token_above_128k_tokens": 0.0000021, 
        "litellm_provider": "gemini",
        "mode": "chat",
        "supports_function_calling": true,
        "supports_vision": true,
        "source": "https://cloud.google.com/vertex-ai/generative-ai/docs/learn/models#foundation_models"
    },
    "gemini/gemini-gemma-2-27b-it": {
        "max_tokens": 8192,
        "max_output_tokens": 8192,
        "input_cost_per_token": 0.00000035, 
        "output_cost_per_token": 0.00000105, 
        "litellm_provider": "gemini",
        "mode": "chat",
        "supports_function_calling": true,
        "supports_vision": true,
        "source": "https://cloud.google.com/vertex-ai/generative-ai/docs/learn/models#foundation_models"
    },
    "gemini/gemini-gemma-2-9b-it": {
        "max_tokens": 8192,
        "max_output_tokens": 8192,
        "input_cost_per_token": 0.00000035, 
        "output_cost_per_token": 0.00000105, 
        "litellm_provider": "gemini",
        "mode": "chat",
        "supports_function_calling": true,
        "supports_vision": true,
        "source": "https://cloud.google.com/vertex-ai/generative-ai/docs/learn/models#foundation_models"
    },
    "command-r": {
        "max_tokens": 4096, 
        "max_input_tokens": 128000,
        "max_output_tokens": 4096,
        "input_cost_per_token": 0.00000050,
        "output_cost_per_token": 0.0000015,
        "litellm_provider": "cohere_chat",
        "mode": "chat",
        "supports_function_calling": true
    },
    "command-light": {
        "max_tokens": 4096, 
        "max_input_tokens": 4096,
        "max_output_tokens": 4096,
        "input_cost_per_token": 0.000015,
        "output_cost_per_token": 0.000015,
        "litellm_provider": "cohere_chat",
        "mode": "chat"
    },
    "command-r-plus": {
        "max_tokens": 4096, 
        "max_input_tokens": 128000,
        "max_output_tokens": 4096,
        "input_cost_per_token": 0.000003,
        "output_cost_per_token": 0.000015,
        "litellm_provider": "cohere_chat",
        "mode": "chat",
        "supports_function_calling": true
    },
    "command-nightly": {
        "max_tokens": 4096, 
        "max_input_tokens": 4096,
        "max_output_tokens": 4096,
        "input_cost_per_token": 0.000015,
        "output_cost_per_token": 0.000015,
        "litellm_provider": "cohere",
        "mode": "completion"
    },
     "command": {
        "max_tokens": 4096, 
        "max_input_tokens": 4096,
        "max_output_tokens": 4096,
        "input_cost_per_token": 0.000015,
        "output_cost_per_token": 0.000015,
        "litellm_provider": "cohere",
        "mode": "completion"
    },
     "command-medium-beta": {
        "max_tokens": 4096, 
        "max_input_tokens": 4096,
        "max_output_tokens": 4096,
        "input_cost_per_token": 0.000015,
        "output_cost_per_token": 0.000015,
        "litellm_provider": "cohere",
        "mode": "completion"
    },
     "command-xlarge-beta": {
        "max_tokens": 4096, 
        "max_input_tokens": 4096,
        "max_output_tokens": 4096,
        "input_cost_per_token": 0.000015,
        "output_cost_per_token": 0.000015,
        "litellm_provider": "cohere",
        "mode": "completion"
    },
    "embed-english-v3.0": {
        "max_tokens": 512, 
        "max_input_tokens": 512,
        "input_cost_per_token": 0.00000010,
        "output_cost_per_token": 0.00000,
        "litellm_provider": "cohere",
        "mode": "embedding"
    },
    "embed-english-light-v3.0": {
        "max_tokens": 512, 
        "max_input_tokens": 512,
        "input_cost_per_token": 0.00000010,
        "output_cost_per_token": 0.00000,
        "litellm_provider": "cohere",
        "mode": "embedding"
    },
    "embed-multilingual-v3.0": {
        "max_tokens": 512, 
        "max_input_tokens": 512,
        "input_cost_per_token": 0.00000010,
        "output_cost_per_token": 0.00000,
        "litellm_provider": "cohere",
        "mode": "embedding"
    },
    "embed-english-v2.0": {
        "max_tokens": 512, 
        "max_input_tokens": 512,
        "input_cost_per_token": 0.00000010,
        "output_cost_per_token": 0.00000,
        "litellm_provider": "cohere",
        "mode": "embedding"
    },
    "embed-english-light-v2.0": {
        "max_tokens": 512, 
        "max_input_tokens": 512,
        "input_cost_per_token": 0.00000010,
        "output_cost_per_token": 0.00000,
        "litellm_provider": "cohere",
        "mode": "embedding"
    },
    "embed-multilingual-v2.0": {
        "max_tokens": 256, 
        "max_input_tokens": 256,
        "input_cost_per_token": 0.00000010,
        "output_cost_per_token": 0.00000,
        "litellm_provider": "cohere",
        "mode": "embedding"
    },
    "replicate/meta/llama-2-13b": {
        "max_tokens": 4096,
        "max_input_tokens": 4096,
        "max_output_tokens": 4096,
        "input_cost_per_token": 0.0000001,
        "output_cost_per_token": 0.0000005,
        "litellm_provider": "replicate",
        "mode": "chat"
    },
    "replicate/meta/llama-2-13b-chat": {
        "max_tokens": 4096,
        "max_input_tokens": 4096,
        "max_output_tokens": 4096,
        "input_cost_per_token": 0.0000001,
        "output_cost_per_token": 0.0000005,
        "litellm_provider": "replicate",
        "mode": "chat"
    },
    "replicate/meta/llama-2-70b": {
        "max_tokens": 4096,
        "max_input_tokens": 4096,
        "max_output_tokens": 4096,
        "input_cost_per_token": 0.00000065,
        "output_cost_per_token": 0.00000275,
        "litellm_provider": "replicate",
        "mode": "chat"
    },
    "replicate/meta/llama-2-70b-chat": {
        "max_tokens": 4096,
        "max_input_tokens": 4096,
        "max_output_tokens": 4096,
        "input_cost_per_token": 0.00000065,
        "output_cost_per_token": 0.00000275,
        "litellm_provider": "replicate",
        "mode": "chat"
    },
    "replicate/meta/llama-2-7b": {
        "max_tokens": 4096,
        "max_input_tokens": 4096,
        "max_output_tokens": 4096,
        "input_cost_per_token": 0.00000005,
        "output_cost_per_token": 0.00000025,
        "litellm_provider": "replicate",
        "mode": "chat"
    },
    "replicate/meta/llama-2-7b-chat": {
        "max_tokens": 4096,
        "max_input_tokens": 4096,
        "max_output_tokens": 4096,
        "input_cost_per_token": 0.00000005,
        "output_cost_per_token": 0.00000025,
        "litellm_provider": "replicate",
        "mode": "chat"
    },
    "replicate/meta/llama-3-70b": {
        "max_tokens": 8192,
        "max_input_tokens": 8192,
        "max_output_tokens": 8192,
        "input_cost_per_token": 0.00000065,
        "output_cost_per_token": 0.00000275,
        "litellm_provider": "replicate",
        "mode": "chat"
    },
    "replicate/meta/llama-3-70b-instruct": {
        "max_tokens": 8192,
        "max_input_tokens": 8192,
        "max_output_tokens": 8192,
        "input_cost_per_token": 0.00000065,
        "output_cost_per_token": 0.00000275,
        "litellm_provider": "replicate",
        "mode": "chat"
    },
    "replicate/meta/llama-3-8b": {
        "max_tokens": 8086,
        "max_input_tokens": 8086,
        "max_output_tokens": 8086,
        "input_cost_per_token": 0.00000005,
        "output_cost_per_token": 0.00000025,
        "litellm_provider": "replicate",
        "mode": "chat"
    },
    "replicate/meta/llama-3-8b-instruct": {
        "max_tokens": 8086,
        "max_input_tokens": 8086,
        "max_output_tokens": 8086,
        "input_cost_per_token": 0.00000005,
        "output_cost_per_token": 0.00000025,
        "litellm_provider": "replicate",
        "mode": "chat"
    },
    "replicate/mistralai/mistral-7b-v0.1": {
        "max_tokens": 4096,
        "max_input_tokens": 4096,
        "max_output_tokens": 4096,
        "input_cost_per_token": 0.00000005,
        "output_cost_per_token": 0.00000025,
        "litellm_provider": "replicate",
        "mode": "chat"
    },
    "replicate/mistralai/mistral-7b-instruct-v0.2": {
        "max_tokens": 4096,
        "max_input_tokens": 4096,
        "max_output_tokens": 4096,
        "input_cost_per_token": 0.00000005,
        "output_cost_per_token": 0.00000025,
        "litellm_provider": "replicate",
        "mode": "chat"
    },
    "replicate/mistralai/mixtral-8x7b-instruct-v0.1": {
        "max_tokens": 4096,
        "max_input_tokens": 4096,
        "max_output_tokens": 4096,
        "input_cost_per_token": 0.0000003,
        "output_cost_per_token": 0.000001,
        "litellm_provider": "replicate",
        "mode": "chat"
    },
    "openrouter/deepseek/deepseek-coder": {
        "max_tokens": 4096,
        "max_input_tokens": 32000,
        "max_output_tokens": 4096,
        "input_cost_per_token": 0.00000014,
        "output_cost_per_token": 0.00000028,
        "litellm_provider": "openrouter",
        "mode": "chat"
    },
    "openrouter/microsoft/wizardlm-2-8x22b:nitro": {
        "max_tokens": 65536,
        "input_cost_per_token": 0.000001,
        "output_cost_per_token": 0.000001,
        "litellm_provider": "openrouter",
        "mode": "chat"
    },
    "openrouter/google/gemini-pro-1.5": {
        "max_tokens": 8192,
        "max_input_tokens": 1000000,
        "max_output_tokens": 8192,
        "input_cost_per_token": 0.0000025,
        "output_cost_per_token": 0.0000075,
        "input_cost_per_image": 0.00265, 
        "litellm_provider": "openrouter",
        "mode": "chat",
        "supports_function_calling": true,
        "supports_vision": true
    },
    "openrouter/mistralai/mixtral-8x22b-instruct": {
        "max_tokens": 65536,
        "input_cost_per_token": 0.00000065,
        "output_cost_per_token": 0.00000065,
        "litellm_provider": "openrouter",
        "mode": "chat"
    },
    "openrouter/cohere/command-r-plus": {
        "max_tokens": 128000,
        "input_cost_per_token": 0.000003,
        "output_cost_per_token": 0.000015,
        "litellm_provider": "openrouter",
        "mode": "chat"
    },
    "openrouter/databricks/dbrx-instruct": {
        "max_tokens": 32768,
        "input_cost_per_token": 0.0000006,
        "output_cost_per_token": 0.0000006,
        "litellm_provider": "openrouter",
        "mode": "chat"
    },
    "openrouter/anthropic/claude-3-haiku": {
        "max_tokens": 200000,
        "input_cost_per_token": 0.00000025,
        "output_cost_per_token": 0.00000125,
        "input_cost_per_image": 0.0004, 
        "litellm_provider": "openrouter",
        "mode": "chat",
        "supports_function_calling": true,
        "supports_vision": true
    },
    "openrouter/anthropic/claude-3-haiku-20240307": {
        "max_tokens": 4096,
        "max_input_tokens": 200000,
        "max_output_tokens": 4096,
        "input_cost_per_token": 0.00000025,
        "output_cost_per_token": 0.00000125,
        "litellm_provider": "openrouter",
        "mode": "chat",
        "supports_function_calling": true,
        "supports_vision": true,
        "tool_use_system_prompt_tokens": 264
    },
    "openrouter/anthropic/claude-3.5-sonnet": {
        "max_tokens": 4096,
        "max_input_tokens": 200000,
        "max_output_tokens": 4096,
        "input_cost_per_token": 0.000003,
        "output_cost_per_token": 0.000015,
        "litellm_provider": "openrouter",
        "mode": "chat",
        "supports_function_calling": true,
        "supports_vision": true,
        "tool_use_system_prompt_tokens": 159,
        "supports_assistant_prefill": true
    },
    "openrouter/anthropic/claude-3.5-sonnet:beta": {
        "max_tokens": 4096,
        "max_input_tokens": 200000,
        "max_output_tokens": 4096,
        "input_cost_per_token": 0.000003,
        "output_cost_per_token": 0.000015,
        "litellm_provider": "openrouter",
        "mode": "chat",
        "supports_function_calling": true,
        "supports_vision": true,
        "tool_use_system_prompt_tokens": 159
    },
    "openrouter/anthropic/claude-3-sonnet": {
        "max_tokens": 200000,
        "input_cost_per_token": 0.000003,
        "output_cost_per_token": 0.000015,
        "input_cost_per_image": 0.0048,  
        "litellm_provider": "openrouter",
        "mode": "chat",
        "supports_function_calling": true,
        "supports_vision": true
    },
    "openrouter/mistralai/mistral-large": {
        "max_tokens": 32000,
        "input_cost_per_token": 0.000008,
        "output_cost_per_token": 0.000024,
        "litellm_provider": "openrouter",
        "mode": "chat"
    },
    "openrouter/cognitivecomputations/dolphin-mixtral-8x7b": {
        "max_tokens": 32769,
        "input_cost_per_token": 0.0000005,
        "output_cost_per_token": 0.0000005,
        "litellm_provider": "openrouter",
        "mode": "chat"
    },
    "openrouter/google/gemini-pro-vision": {
        "max_tokens": 45875,
        "input_cost_per_token": 0.000000125,
        "output_cost_per_token": 0.000000375,
        "input_cost_per_image": 0.0025,  
        "litellm_provider": "openrouter",
        "mode": "chat",
        "supports_function_calling": true,
        "supports_vision": true
    },
    "openrouter/fireworks/firellava-13b": {
        "max_tokens": 4096,
        "input_cost_per_token": 0.0000002,
        "output_cost_per_token": 0.0000002,
        "litellm_provider": "openrouter",
        "mode": "chat"
    },
    "openrouter/meta-llama/llama-3-8b-instruct:free": {
        "max_tokens": 8192,
        "input_cost_per_token": 0.0,
        "output_cost_per_token": 0.0,
        "litellm_provider": "openrouter",
        "mode": "chat"
    },
    "openrouter/meta-llama/llama-3-8b-instruct:extended": {
        "max_tokens": 16384,
        "input_cost_per_token": 0.000000225,
        "output_cost_per_token": 0.00000225,
        "litellm_provider": "openrouter",
        "mode": "chat"
    },
    "openrouter/meta-llama/llama-3-70b-instruct:nitro": {
        "max_tokens": 8192,
        "input_cost_per_token": 0.0000009,
        "output_cost_per_token": 0.0000009,
        "litellm_provider": "openrouter",
        "mode": "chat"
    },
    "openrouter/meta-llama/llama-3-70b-instruct": {
        "max_tokens": 8192,
        "input_cost_per_token": 0.00000059,
        "output_cost_per_token": 0.00000079,
        "litellm_provider": "openrouter",
        "mode": "chat"
    },
    "openrouter/openai/gpt-4o": {
        "max_tokens": 4096,
        "max_input_tokens": 128000,
        "max_output_tokens": 4096,
        "input_cost_per_token": 0.000005,
        "output_cost_per_token": 0.000015,
        "litellm_provider": "openrouter",
        "mode": "chat",
        "supports_function_calling": true,
        "supports_parallel_function_calling": true,
        "supports_vision": true
    },
    "openrouter/openai/gpt-4o-2024-05-13": {
        "max_tokens": 4096,
        "max_input_tokens": 128000,
        "max_output_tokens": 4096,
        "input_cost_per_token": 0.000005,
        "output_cost_per_token": 0.000015,
        "litellm_provider": "openrouter",
        "mode": "chat",
        "supports_function_calling": true,
        "supports_parallel_function_calling": true,
        "supports_vision": true
    },
    "openrouter/openai/gpt-4-vision-preview": {
        "max_tokens": 130000,
        "input_cost_per_token": 0.00001,
        "output_cost_per_token": 0.00003,
        "input_cost_per_image": 0.01445, 
        "litellm_provider": "openrouter",
        "mode": "chat",
        "supports_function_calling": true,
        "supports_vision": true
    },
    "openrouter/openai/gpt-3.5-turbo": {
        "max_tokens": 4095,
        "input_cost_per_token": 0.0000015,
        "output_cost_per_token": 0.000002,
        "litellm_provider": "openrouter",
        "mode": "chat"
    },
    "openrouter/openai/gpt-3.5-turbo-16k": {
        "max_tokens": 16383,
        "input_cost_per_token": 0.000003,
        "output_cost_per_token": 0.000004,
        "litellm_provider": "openrouter",
        "mode": "chat"
    },
    "openrouter/openai/gpt-4": {
        "max_tokens": 8192,
        "input_cost_per_token": 0.00003,
        "output_cost_per_token": 0.00006,
        "litellm_provider": "openrouter",
        "mode": "chat"
    },
    "openrouter/anthropic/claude-instant-v1": {
        "max_tokens": 100000,
        "max_output_tokens": 8191,
        "input_cost_per_token": 0.00000163,
        "output_cost_per_token": 0.00000551,
        "litellm_provider": "openrouter",
        "mode": "chat"
    },
    "openrouter/anthropic/claude-2": {
        "max_tokens": 100000,
        "max_output_tokens": 8191,
        "input_cost_per_token": 0.00001102,
        "output_cost_per_token": 0.00003268,
        "litellm_provider": "openrouter",
        "mode": "chat"
    },
    "openrouter/anthropic/claude-3-opus": {
        "max_tokens": 4096,
        "max_input_tokens": 200000,
        "max_output_tokens": 4096,
        "input_cost_per_token": 0.000015,
        "output_cost_per_token": 0.000075,
        "litellm_provider": "openrouter",
        "mode": "chat",
        "supports_function_calling": true,
        "supports_vision": true,
        "tool_use_system_prompt_tokens": 395
    },
    "openrouter/google/palm-2-chat-bison": {
        "max_tokens": 25804,
        "input_cost_per_token": 0.0000005,
        "output_cost_per_token": 0.0000005,
        "litellm_provider": "openrouter",
        "mode": "chat"
    },
    "openrouter/google/palm-2-codechat-bison": {
        "max_tokens": 20070,
        "input_cost_per_token": 0.0000005,
        "output_cost_per_token": 0.0000005,
        "litellm_provider": "openrouter",
        "mode": "chat"
    },
    "openrouter/meta-llama/llama-2-13b-chat": {
        "max_tokens": 4096,
        "input_cost_per_token": 0.0000002,
        "output_cost_per_token": 0.0000002,
        "litellm_provider": "openrouter",
        "mode": "chat"
    },
    "openrouter/meta-llama/llama-2-70b-chat": {
        "max_tokens": 4096,
        "input_cost_per_token": 0.0000015,
        "output_cost_per_token": 0.0000015,
        "litellm_provider": "openrouter",
        "mode": "chat"
    },
    "openrouter/meta-llama/codellama-34b-instruct": {
        "max_tokens": 8192,
        "input_cost_per_token": 0.0000005,
        "output_cost_per_token": 0.0000005,
        "litellm_provider": "openrouter",
        "mode": "chat"
    },
    "openrouter/nousresearch/nous-hermes-llama2-13b": {
        "max_tokens": 4096,
        "input_cost_per_token": 0.0000002,
        "output_cost_per_token": 0.0000002,
        "litellm_provider": "openrouter",
        "mode": "chat"
    },
    "openrouter/mancer/weaver": {
        "max_tokens": 8000,
        "input_cost_per_token": 0.000005625,
        "output_cost_per_token": 0.000005625,
        "litellm_provider": "openrouter",
        "mode": "chat"
    },
    "openrouter/gryphe/mythomax-l2-13b": {
        "max_tokens": 8192,
        "input_cost_per_token": 0.000001875,
        "output_cost_per_token": 0.000001875,
        "litellm_provider": "openrouter",
        "mode": "chat"
    },
    "openrouter/jondurbin/airoboros-l2-70b-2.1": {
        "max_tokens": 4096,
        "input_cost_per_token": 0.000013875,
        "output_cost_per_token": 0.000013875,
        "litellm_provider": "openrouter",
        "mode": "chat"
    },
    "openrouter/undi95/remm-slerp-l2-13b": {
        "max_tokens": 6144,
        "input_cost_per_token": 0.000001875,
        "output_cost_per_token": 0.000001875,
        "litellm_provider": "openrouter",
        "mode": "chat"
    },
    "openrouter/pygmalionai/mythalion-13b": {
        "max_tokens": 4096,
        "input_cost_per_token": 0.000001875,
        "output_cost_per_token": 0.000001875,
        "litellm_provider": "openrouter",
        "mode": "chat"
    },
    "openrouter/mistralai/mistral-7b-instruct": {
        "max_tokens": 8192,
        "input_cost_per_token": 0.00000013,
        "output_cost_per_token": 0.00000013,
        "litellm_provider": "openrouter",
        "mode": "chat"
    },
    "openrouter/mistralai/mistral-7b-instruct:free": {
        "max_tokens": 8192,
        "input_cost_per_token": 0.0,
        "output_cost_per_token": 0.0,
        "litellm_provider": "openrouter",
        "mode": "chat"
    },
    "j2-ultra": {
        "max_tokens": 8192,
        "max_input_tokens": 8192,
        "max_output_tokens": 8192,
        "input_cost_per_token": 0.000015,
        "output_cost_per_token": 0.000015,
        "litellm_provider": "ai21",
        "mode": "completion"
    },
    "jamba-1.5-mini@001": {
        "max_tokens": 256000,
        "max_input_tokens": 256000,
        "max_output_tokens": 256000,
        "input_cost_per_token": 0.0000002,
        "output_cost_per_token": 0.0000004,
        "litellm_provider": "ai21",
        "mode": "chat"
    },
    "jamba-1.5-large@001": {
        "max_tokens": 256000,
        "max_input_tokens": 256000,
        "max_output_tokens": 256000,
        "input_cost_per_token": 0.000002,
        "output_cost_per_token": 0.000008,
        "litellm_provider": "ai21",
        "mode": "chat"
    },
    "jamba-1.5": {
        "max_tokens": 256000,
        "max_input_tokens": 256000,
        "max_output_tokens": 256000,
        "input_cost_per_token": 0.0000002,
        "output_cost_per_token": 0.0000004,
        "litellm_provider": "ai21",
        "mode": "chat"
    },
    "jamba-1.5-mini": {
        "max_tokens": 256000,
        "max_input_tokens": 256000,
        "max_output_tokens": 256000,
        "input_cost_per_token": 0.0000002,
        "output_cost_per_token": 0.0000004,
        "litellm_provider": "ai21",
        "mode": "chat"
    },
    "jamba-1.5-large": {
        "max_tokens": 256000,
        "max_input_tokens": 256000,
        "max_output_tokens": 256000,
        "input_cost_per_token": 0.000002,
        "output_cost_per_token": 0.000008,
        "litellm_provider": "ai21",
        "mode": "chat"
    },
    "j2-mid": {
        "max_tokens": 8192,
        "max_input_tokens": 8192,
        "max_output_tokens": 8192,
        "input_cost_per_token": 0.00001,
        "output_cost_per_token": 0.00001,
        "litellm_provider": "ai21",
        "mode": "completion"
    },
    "j2-light": {
        "max_tokens": 8192,
        "max_input_tokens": 8192,
        "max_output_tokens": 8192,
        "input_cost_per_token": 0.000003,
        "output_cost_per_token": 0.000003,
        "litellm_provider": "ai21",
        "mode": "completion"
    },
    "dolphin": {
        "max_tokens": 16384,
        "max_input_tokens": 16384,
        "max_output_tokens": 16384,
        "input_cost_per_token": 0.0000005,
        "output_cost_per_token": 0.0000005,
        "litellm_provider": "nlp_cloud",
        "mode": "completion"
    },
    "chatdolphin": {
        "max_tokens": 16384,
        "max_input_tokens": 16384,
        "max_output_tokens": 16384,
        "input_cost_per_token": 0.0000005,
        "output_cost_per_token": 0.0000005,
        "litellm_provider": "nlp_cloud",
        "mode": "chat"
    },
    "luminous-base": {
        "max_tokens": 2048, 
        "input_cost_per_token": 0.00003,
        "output_cost_per_token": 0.000033,
        "litellm_provider": "aleph_alpha",
        "mode": "completion"
    },
    "luminous-base-control": {
        "max_tokens": 2048, 
        "input_cost_per_token": 0.0000375,
        "output_cost_per_token": 0.00004125,
        "litellm_provider": "aleph_alpha",
        "mode": "chat"
    },
    "luminous-extended": {
        "max_tokens": 2048, 
        "input_cost_per_token": 0.000045,
        "output_cost_per_token": 0.0000495,
        "litellm_provider": "aleph_alpha",
        "mode": "completion"
    },
    "luminous-extended-control": {
        "max_tokens": 2048, 
        "input_cost_per_token": 0.00005625,
        "output_cost_per_token": 0.000061875,
        "litellm_provider": "aleph_alpha",
        "mode": "chat"
    },
    "luminous-supreme": {
        "max_tokens": 2048, 
        "input_cost_per_token": 0.000175,
        "output_cost_per_token": 0.0001925,
        "litellm_provider": "aleph_alpha",
        "mode": "completion"
    },
    "luminous-supreme-control": {
        "max_tokens": 2048, 
        "input_cost_per_token": 0.00021875,
        "output_cost_per_token": 0.000240625,
        "litellm_provider": "aleph_alpha",
        "mode": "chat"
    },
    "ai21.j2-mid-v1": {
        "max_tokens": 8191, 
        "max_input_tokens": 8191, 
        "max_output_tokens": 8191, 
        "input_cost_per_token": 0.0000125,
        "output_cost_per_token": 0.0000125,
        "litellm_provider": "bedrock",
        "mode": "chat"
    },
    "ai21.j2-ultra-v1": {
        "max_tokens": 8191, 
        "max_input_tokens": 8191, 
        "max_output_tokens": 8191, 
        "input_cost_per_token": 0.0000188,
        "output_cost_per_token": 0.0000188,
        "litellm_provider": "bedrock",
        "mode": "chat"
    },
    "ai21.jamba-instruct-v1:0": {
        "max_tokens": 4096,
        "max_input_tokens": 70000,
        "max_output_tokens": 4096,
        "input_cost_per_token": 0.0000005,
        "output_cost_per_token": 0.0000007,
        "litellm_provider": "bedrock",
        "mode": "chat",
        "supports_system_messages": true
    },
    "amazon.titan-text-lite-v1": {
        "max_tokens": 4000, 
        "max_input_tokens": 42000,
        "max_output_tokens": 4000, 
        "input_cost_per_token": 0.0000003,
        "output_cost_per_token": 0.0000004,
        "litellm_provider": "bedrock",
        "mode": "chat"
    },
    "amazon.titan-text-express-v1": {
        "max_tokens": 8000, 
        "max_input_tokens": 42000,
        "max_output_tokens": 8000, 
        "input_cost_per_token": 0.0000013,
        "output_cost_per_token": 0.0000017,
        "litellm_provider": "bedrock",
        "mode": "chat"
    },
    "amazon.titan-text-premier-v1:0": {
        "max_tokens": 32000, 
        "max_input_tokens": 42000,
        "max_output_tokens": 32000, 
        "input_cost_per_token": 0.0000005,
        "output_cost_per_token": 0.0000015,
        "litellm_provider": "bedrock",
        "mode": "chat"
    },
    "amazon.titan-embed-text-v1": {
        "max_tokens": 8192, 
        "max_input_tokens": 8192, 
        "output_vector_size": 1536,
        "input_cost_per_token": 0.0000001,
        "output_cost_per_token": 0.0,
        "litellm_provider": "bedrock", 
        "mode": "embedding"
    },
    "amazon.titan-embed-text-v2:0": {
        "max_tokens": 8192, 
        "max_input_tokens": 8192, 
        "output_vector_size": 1024,
        "input_cost_per_token": 0.0000002,
        "output_cost_per_token": 0.0,
        "litellm_provider": "bedrock", 
        "mode": "embedding"
    },
    "mistral.mistral-7b-instruct-v0:2": {
        "max_tokens": 8191,
        "max_input_tokens": 32000,
        "max_output_tokens": 8191,
        "input_cost_per_token": 0.00000015,
        "output_cost_per_token": 0.0000002,
        "litellm_provider": "bedrock",
        "mode": "chat"
    },
    "mistral.mixtral-8x7b-instruct-v0:1": {
        "max_tokens": 8191,
        "max_input_tokens": 32000,
        "max_output_tokens": 8191,
        "input_cost_per_token": 0.00000045,
        "output_cost_per_token": 0.0000007,
        "litellm_provider": "bedrock",
        "mode": "chat"
    },
    "mistral.mistral-large-2402-v1:0": {
        "max_tokens": 8191,
        "max_input_tokens": 32000,
        "max_output_tokens": 8191,
        "input_cost_per_token": 0.000008,
        "output_cost_per_token": 0.000024,
        "litellm_provider": "bedrock",
        "mode": "chat",
        "supports_function_calling": true
    },
    "mistral.mistral-large-2407-v1:0": {
        "max_tokens": 8191,
        "max_input_tokens": 128000,
        "max_output_tokens": 8191,
        "input_cost_per_token": 0.000003,
        "output_cost_per_token": 0.000009,
        "litellm_provider": "bedrock",
        "mode": "chat",
        "supports_function_calling": true
    },
    "mistral.mistral-small-2402-v1:0": {
        "max_tokens": 8191,
        "max_input_tokens": 32000,
        "max_output_tokens": 8191,
        "input_cost_per_token": 0.000001,
        "output_cost_per_token": 0.000003,
        "litellm_provider": "bedrock",
        "mode": "chat",
        "supports_function_calling": true
    },
    "bedrock/us-west-2/mistral.mixtral-8x7b-instruct-v0:1": {
        "max_tokens": 8191,
        "max_input_tokens": 32000,
        "max_output_tokens": 8191,
        "input_cost_per_token": 0.00000045,
        "output_cost_per_token": 0.0000007,
        "litellm_provider": "bedrock",
        "mode": "chat"
    },
    "bedrock/us-east-1/mistral.mixtral-8x7b-instruct-v0:1": {
        "max_tokens": 8191,
        "max_input_tokens": 32000,
        "max_output_tokens": 8191,
        "input_cost_per_token": 0.00000045,
        "output_cost_per_token": 0.0000007,
        "litellm_provider": "bedrock",
        "mode": "chat"
    },
    "bedrock/eu-west-3/mistral.mixtral-8x7b-instruct-v0:1": {
        "max_tokens": 8191,
        "max_input_tokens": 32000,
        "max_output_tokens": 8191,
        "input_cost_per_token": 0.00000059,
        "output_cost_per_token": 0.00000091,
        "litellm_provider": "bedrock",
        "mode": "chat"
    },
    "bedrock/us-west-2/mistral.mistral-7b-instruct-v0:2": {
        "max_tokens": 8191,
        "max_input_tokens": 32000,
        "max_output_tokens": 8191,
        "input_cost_per_token": 0.00000015,
        "output_cost_per_token": 0.0000002,
        "litellm_provider": "bedrock",
        "mode": "chat"
    },
    "bedrock/us-east-1/mistral.mistral-7b-instruct-v0:2": {
        "max_tokens": 8191,
        "max_input_tokens": 32000,
        "max_output_tokens": 8191,
        "input_cost_per_token": 0.00000015,
        "output_cost_per_token": 0.0000002,
        "litellm_provider": "bedrock",
        "mode": "chat"
    },
    "bedrock/eu-west-3/mistral.mistral-7b-instruct-v0:2": {
        "max_tokens": 8191,
        "max_input_tokens": 32000,
        "max_output_tokens": 8191,
        "input_cost_per_token": 0.0000002,
        "output_cost_per_token": 0.00000026,
        "litellm_provider": "bedrock",
        "mode": "chat"
    },
    "bedrock/us-east-1/mistral.mistral-large-2402-v1:0": {
        "max_tokens": 8191,
        "max_input_tokens": 32000,
        "max_output_tokens": 8191,
        "input_cost_per_token": 0.000008,
        "output_cost_per_token": 0.000024,
        "litellm_provider": "bedrock",
        "mode": "chat"
    },
    "bedrock/us-west-2/mistral.mistral-large-2402-v1:0": {
        "max_tokens": 8191,
        "max_input_tokens": 32000,
        "max_output_tokens": 8191,
        "input_cost_per_token": 0.000008,
        "output_cost_per_token": 0.000024,
        "litellm_provider": "bedrock",
        "mode": "chat",
        "supports_function_calling": true
    },
    "bedrock/eu-west-3/mistral.mistral-large-2402-v1:0": {
        "max_tokens": 8191,
        "max_input_tokens": 32000,
        "max_output_tokens": 8191,
        "input_cost_per_token": 0.0000104,
        "output_cost_per_token": 0.0000312,
        "litellm_provider": "bedrock",
        "mode": "chat",
        "supports_function_calling": true
    },
    "anthropic.claude-3-sonnet-20240229-v1:0": {
        "max_tokens": 4096, 
        "max_input_tokens": 200000,
        "max_output_tokens": 4096,
        "input_cost_per_token": 0.000003,
        "output_cost_per_token": 0.000015,
        "litellm_provider": "bedrock",
        "mode": "chat",
        "supports_function_calling": true,
        "supports_vision": true
    },
    "anthropic.claude-3-5-sonnet-20240620-v1:0": {
        "max_tokens": 4096, 
        "max_input_tokens": 200000,
        "max_output_tokens": 4096,
        "input_cost_per_token": 0.000003,
        "output_cost_per_token": 0.000015,
        "litellm_provider": "bedrock",
        "mode": "chat",
        "supports_function_calling": true,
        "supports_vision": true
    },
    "anthropic.claude-3-haiku-20240307-v1:0": {
        "max_tokens": 4096, 
        "max_input_tokens": 200000,
        "max_output_tokens": 4096,
        "input_cost_per_token": 0.00000025,
        "output_cost_per_token": 0.00000125,
        "litellm_provider": "bedrock",
        "mode": "chat",
        "supports_function_calling": true,
        "supports_vision": true
    },
    "anthropic.claude-3-opus-20240229-v1:0": {
        "max_tokens": 4096,
        "max_input_tokens": 200000,
        "max_output_tokens": 4096,
        "input_cost_per_token": 0.000015,
        "output_cost_per_token": 0.000075,
        "litellm_provider": "bedrock",
        "mode": "chat",
        "supports_function_calling": true,
        "supports_vision": true
    },
    "us.anthropic.claude-3-sonnet-20240229-v1:0": {
        "max_tokens": 4096,
        "max_input_tokens": 200000,
        "max_output_tokens": 4096,
        "input_cost_per_token": 0.000003,
        "output_cost_per_token": 0.000015,
        "litellm_provider": "bedrock",
        "mode": "chat",
        "supports_function_calling": true,
        "supports_vision": true
    },
    "us.anthropic.claude-3-5-sonnet-20240620-v1:0": {
        "max_tokens": 4096,
        "max_input_tokens": 200000,
        "max_output_tokens": 4096,
        "input_cost_per_token": 0.000003,
        "output_cost_per_token": 0.000015,
        "litellm_provider": "bedrock",
        "mode": "chat",
        "supports_function_calling": true,
        "supports_vision": true
    },
    "us.anthropic.claude-3-haiku-20240307-v1:0": {
        "max_tokens": 4096,
        "max_input_tokens": 200000,
        "max_output_tokens": 4096,
        "input_cost_per_token": 0.00000025,
        "output_cost_per_token": 0.00000125,
        "litellm_provider": "bedrock",
        "mode": "chat",
        "supports_function_calling": true,
        "supports_vision": true
    },
    "us.anthropic.claude-3-opus-20240229-v1:0": {
        "max_tokens": 4096,
        "max_input_tokens": 200000,
        "max_output_tokens": 4096,
        "input_cost_per_token": 0.000015,
        "output_cost_per_token": 0.000075,
        "litellm_provider": "bedrock",
        "mode": "chat",
        "supports_function_calling": true,
        "supports_vision": true
    },
    "eu.anthropic.claude-3-sonnet-20240229-v1:0": {
        "max_tokens": 4096,
        "max_input_tokens": 200000,
        "max_output_tokens": 4096,
        "input_cost_per_token": 0.000003,
        "output_cost_per_token": 0.000015,
        "litellm_provider": "bedrock",
        "mode": "chat",
        "supports_function_calling": true,
        "supports_vision": true
    },
    "eu.anthropic.claude-3-5-sonnet-20240620-v1:0": {
        "max_tokens": 4096,
        "max_input_tokens": 200000,
        "max_output_tokens": 4096,
        "input_cost_per_token": 0.000003,
        "output_cost_per_token": 0.000015,
        "litellm_provider": "bedrock",
        "mode": "chat",
        "supports_function_calling": true,
        "supports_vision": true
    },
    "eu.anthropic.claude-3-haiku-20240307-v1:0": {
        "max_tokens": 4096,
        "max_input_tokens": 200000,
        "max_output_tokens": 4096,
        "input_cost_per_token": 0.00000025,
        "output_cost_per_token": 0.00000125,
        "litellm_provider": "bedrock",
        "mode": "chat",
        "supports_function_calling": true,
        "supports_vision": true
    },
    "eu.anthropic.claude-3-opus-20240229-v1:0": {
        "max_tokens": 4096,
        "max_input_tokens": 200000,
        "max_output_tokens": 4096,
        "input_cost_per_token": 0.000015,
        "output_cost_per_token": 0.000075,
        "litellm_provider": "bedrock",
        "mode": "chat",
        "supports_function_calling": true,
        "supports_vision": true
    },
    "anthropic.claude-v1": {
        "max_tokens": 8191, 
        "max_input_tokens": 100000,
        "max_output_tokens": 8191,
        "input_cost_per_token": 0.000008,
        "output_cost_per_token": 0.000024,
        "litellm_provider": "bedrock",
        "mode": "chat"
    },
    "bedrock/us-east-1/anthropic.claude-v1": {
        "max_tokens": 8191, 
        "max_input_tokens": 100000,
        "max_output_tokens": 8191,
        "input_cost_per_token": 0.000008,
        "output_cost_per_token": 0.000024,
        "litellm_provider": "bedrock",
        "mode": "chat"
    },
    "bedrock/us-west-2/anthropic.claude-v1": {
        "max_tokens": 8191, 
        "max_input_tokens": 100000,
        "max_output_tokens": 8191,
        "input_cost_per_token": 0.000008,
        "output_cost_per_token": 0.000024,
        "litellm_provider": "bedrock",
        "mode": "chat"
    },
    "bedrock/ap-northeast-1/anthropic.claude-v1": {
        "max_tokens": 8191, 
        "max_input_tokens": 100000,
        "max_output_tokens": 8191,
        "input_cost_per_token": 0.000008,
        "output_cost_per_token": 0.000024,
        "litellm_provider": "bedrock",
        "mode": "chat"
    },
    "bedrock/ap-northeast-1/1-month-commitment/anthropic.claude-v1": {
        "max_tokens": 8191, 
        "max_input_tokens": 100000,
        "max_output_tokens": 8191,
        "input_cost_per_second": 0.0455,
        "output_cost_per_second": 0.0455,
        "litellm_provider": "bedrock",
        "mode": "chat"
    },
    "bedrock/ap-northeast-1/6-month-commitment/anthropic.claude-v1": {
        "max_tokens": 8191, 
        "max_input_tokens": 100000,
        "max_output_tokens": 8191,
        "input_cost_per_second": 0.02527,
        "output_cost_per_second": 0.02527,
        "litellm_provider": "bedrock",
        "mode": "chat"
    },
    "bedrock/eu-central-1/anthropic.claude-v1": {
        "max_tokens": 8191, 
        "max_input_tokens": 100000,
        "max_output_tokens": 8191, 
        "input_cost_per_token": 0.000008,
        "output_cost_per_token": 0.000024,
        "litellm_provider": "bedrock",
        "mode": "chat"
    },
    "bedrock/eu-central-1/1-month-commitment/anthropic.claude-v1": {
        "max_tokens": 8191, 
        "max_input_tokens": 100000,
        "max_output_tokens": 8191, 
        "input_cost_per_second": 0.0415,
        "output_cost_per_second": 0.0415,
        "litellm_provider": "bedrock",
        "mode": "chat"
    },
    "bedrock/eu-central-1/6-month-commitment/anthropic.claude-v1": {
        "max_tokens": 8191, 
        "max_input_tokens": 100000,
        "max_output_tokens": 8191, 
        "input_cost_per_second": 0.02305,
        "output_cost_per_second": 0.02305,
        "litellm_provider": "bedrock",
        "mode": "chat"
    },
    "bedrock/us-east-1/1-month-commitment/anthropic.claude-v1": {
        "max_tokens": 8191, 
        "max_input_tokens": 100000,
        "max_output_tokens": 8191, 
        "input_cost_per_second": 0.0175,
        "output_cost_per_second": 0.0175,
        "litellm_provider": "bedrock",
        "mode": "chat"
    },
    "bedrock/us-east-1/6-month-commitment/anthropic.claude-v1": {
        "max_tokens": 8191, 
        "max_input_tokens": 100000,
        "max_output_tokens": 8191, 
        "input_cost_per_second": 0.00972,
        "output_cost_per_second": 0.00972,
        "litellm_provider": "bedrock",
        "mode": "chat"
    },
    "bedrock/us-west-2/1-month-commitment/anthropic.claude-v1": {
        "max_tokens": 8191, 
        "max_input_tokens": 100000,
        "max_output_tokens": 8191, 
        "input_cost_per_second": 0.0175,
        "output_cost_per_second": 0.0175,
        "litellm_provider": "bedrock",
        "mode": "chat"
    },
    "bedrock/us-west-2/6-month-commitment/anthropic.claude-v1": {
        "max_tokens": 8191, 
        "max_input_tokens": 100000,
        "max_output_tokens": 8191, 
        "input_cost_per_second": 0.00972,
        "output_cost_per_second": 0.00972,
        "litellm_provider": "bedrock",
        "mode": "chat"
    },
    "anthropic.claude-v2": {
        "max_tokens": 8191, 
        "max_input_tokens": 100000,
        "max_output_tokens": 8191, 
        "input_cost_per_token": 0.000008,
        "output_cost_per_token": 0.000024,
        "litellm_provider": "bedrock",
        "mode": "chat"
    },
    "bedrock/us-east-1/anthropic.claude-v2": {
        "max_tokens": 8191, 
        "max_input_tokens": 100000,
        "max_output_tokens": 8191, 
        "input_cost_per_token": 0.000008,
        "output_cost_per_token": 0.000024,
        "litellm_provider": "bedrock",
        "mode": "chat"
    },
    "bedrock/us-west-2/anthropic.claude-v2": {
        "max_tokens": 8191, 
        "max_input_tokens": 100000,
        "max_output_tokens": 8191, 
        "input_cost_per_token": 0.000008,
        "output_cost_per_token": 0.000024,
        "litellm_provider": "bedrock",
        "mode": "chat"
    },
    "bedrock/ap-northeast-1/anthropic.claude-v2": {
        "max_tokens": 8191, 
        "max_input_tokens": 100000,
        "max_output_tokens": 8191, 
        "input_cost_per_token": 0.000008,
        "output_cost_per_token": 0.000024,
        "litellm_provider": "bedrock",
        "mode": "chat"
    },
    "bedrock/ap-northeast-1/1-month-commitment/anthropic.claude-v2": {
        "max_tokens": 8191, 
        "max_input_tokens": 100000,
        "max_output_tokens": 8191, 
        "input_cost_per_second": 0.0455,
        "output_cost_per_second": 0.0455,
        "litellm_provider": "bedrock",
        "mode": "chat"
    },
    "bedrock/ap-northeast-1/6-month-commitment/anthropic.claude-v2": {
        "max_tokens": 8191, 
        "max_input_tokens": 100000,
        "max_output_tokens": 8191, 
        "input_cost_per_second": 0.02527,
        "output_cost_per_second": 0.02527,
        "litellm_provider": "bedrock",
        "mode": "chat"
    },
    "bedrock/eu-central-1/anthropic.claude-v2": {
        "max_tokens": 8191, 
        "max_input_tokens": 100000,
        "max_output_tokens": 8191, 
        "input_cost_per_token": 0.000008,
        "output_cost_per_token": 0.000024,
        "litellm_provider": "bedrock",
        "mode": "chat"
    },
    "bedrock/eu-central-1/1-month-commitment/anthropic.claude-v2": {
        "max_tokens": 8191, 
        "max_input_tokens": 100000,
        "max_output_tokens": 8191, 
        "input_cost_per_second": 0.0415,
        "output_cost_per_second": 0.0415,
        "litellm_provider": "bedrock",
        "mode": "chat"
    },
    "bedrock/eu-central-1/6-month-commitment/anthropic.claude-v2": {
        "max_tokens": 8191, 
        "max_input_tokens": 100000,
        "max_output_tokens": 8191, 
        "input_cost_per_second": 0.02305,
        "output_cost_per_second": 0.02305,
        "litellm_provider": "bedrock",
        "mode": "chat"
    },
    "bedrock/us-east-1/1-month-commitment/anthropic.claude-v2": {
        "max_tokens": 8191, 
        "max_input_tokens": 100000,
        "max_output_tokens": 8191, 
        "input_cost_per_second": 0.0175,
        "output_cost_per_second": 0.0175,
        "litellm_provider": "bedrock",
        "mode": "chat"
    },
    "bedrock/us-east-1/6-month-commitment/anthropic.claude-v2": {
        "max_tokens": 8191, 
        "max_input_tokens": 100000,
        "max_output_tokens": 8191, 
        "input_cost_per_second": 0.00972,
        "output_cost_per_second": 0.00972,
        "litellm_provider": "bedrock",
        "mode": "chat"
    },
    "bedrock/us-west-2/1-month-commitment/anthropic.claude-v2": {
        "max_tokens": 8191, 
        "max_input_tokens": 100000,
        "max_output_tokens": 8191, 
        "input_cost_per_second": 0.0175,
        "output_cost_per_second": 0.0175,
        "litellm_provider": "bedrock",
        "mode": "chat"
    },
    "bedrock/us-west-2/6-month-commitment/anthropic.claude-v2": {
        "max_tokens": 8191, 
        "max_input_tokens": 100000,
        "max_output_tokens": 8191, 
        "input_cost_per_second": 0.00972,
        "output_cost_per_second": 0.00972,
        "litellm_provider": "bedrock",
        "mode": "chat"
    },
    "anthropic.claude-v2:1": {
        "max_tokens": 8191, 
        "max_input_tokens": 100000, 
        "max_output_tokens": 8191,
        "input_cost_per_token": 0.000008,
        "output_cost_per_token": 0.000024,
        "litellm_provider": "bedrock",
        "mode": "chat"
    },
    "bedrock/us-east-1/anthropic.claude-v2:1": {
        "max_tokens": 8191, 
        "max_input_tokens": 100000, 
        "max_output_tokens": 8191,
        "input_cost_per_token": 0.000008,
        "output_cost_per_token": 0.000024,
        "litellm_provider": "bedrock",
        "mode": "chat"
    },
    "bedrock/us-west-2/anthropic.claude-v2:1": {
        "max_tokens": 8191, 
        "max_input_tokens": 100000, 
        "max_output_tokens": 8191,
        "input_cost_per_token": 0.000008,
        "output_cost_per_token": 0.000024,
        "litellm_provider": "bedrock",
        "mode": "chat"
    },
    "bedrock/ap-northeast-1/anthropic.claude-v2:1": {
        "max_tokens": 8191, 
        "max_input_tokens": 100000, 
        "max_output_tokens": 8191,
        "input_cost_per_token": 0.000008,
        "output_cost_per_token": 0.000024,
        "litellm_provider": "bedrock",
        "mode": "chat"
    },
    "bedrock/ap-northeast-1/1-month-commitment/anthropic.claude-v2:1": {
        "max_tokens": 8191, 
        "max_input_tokens": 100000, 
        "max_output_tokens": 8191,
        "input_cost_per_second": 0.0455,
        "output_cost_per_second": 0.0455,
        "litellm_provider": "bedrock",
        "mode": "chat"
    },
    "bedrock/ap-northeast-1/6-month-commitment/anthropic.claude-v2:1": {
        "max_tokens": 8191, 
        "max_input_tokens": 100000, 
        "max_output_tokens": 8191,
        "input_cost_per_second": 0.02527,
        "output_cost_per_second": 0.02527,
        "litellm_provider": "bedrock",
        "mode": "chat"
    },
    "bedrock/eu-central-1/anthropic.claude-v2:1": {
        "max_tokens": 8191, 
        "max_input_tokens": 100000, 
        "max_output_tokens": 8191,
        "input_cost_per_token": 0.000008,
        "output_cost_per_token": 0.000024,
        "litellm_provider": "bedrock",
        "mode": "chat"
    },
    "bedrock/eu-central-1/1-month-commitment/anthropic.claude-v2:1": {
        "max_tokens": 8191, 
        "max_input_tokens": 100000, 
        "max_output_tokens": 8191,
        "input_cost_per_second": 0.0415,
        "output_cost_per_second": 0.0415,
        "litellm_provider": "bedrock",
        "mode": "chat"
    },
    "bedrock/eu-central-1/6-month-commitment/anthropic.claude-v2:1": {
        "max_tokens": 8191, 
        "max_input_tokens": 100000, 
        "max_output_tokens": 8191,
        "input_cost_per_second": 0.02305,
        "output_cost_per_second": 0.02305,
        "litellm_provider": "bedrock",
        "mode": "chat"
    },
    "bedrock/us-east-1/1-month-commitment/anthropic.claude-v2:1": {
        "max_tokens": 8191, 
        "max_input_tokens": 100000, 
        "max_output_tokens": 8191,
        "input_cost_per_second": 0.0175,
        "output_cost_per_second": 0.0175,
        "litellm_provider": "bedrock",
        "mode": "chat"
    },
    "bedrock/us-east-1/6-month-commitment/anthropic.claude-v2:1": {
        "max_tokens": 8191, 
        "max_input_tokens": 100000, 
        "max_output_tokens": 8191,
        "input_cost_per_second": 0.00972,
        "output_cost_per_second": 0.00972,
        "litellm_provider": "bedrock",
        "mode": "chat"
    },
    "bedrock/us-west-2/1-month-commitment/anthropic.claude-v2:1": {
        "max_tokens": 8191, 
        "max_input_tokens": 100000, 
        "max_output_tokens": 8191,
        "input_cost_per_second": 0.0175,
        "output_cost_per_second": 0.0175,
        "litellm_provider": "bedrock",
        "mode": "chat"
    },
    "bedrock/us-west-2/6-month-commitment/anthropic.claude-v2:1": {
        "max_tokens": 8191, 
        "max_input_tokens": 100000, 
        "max_output_tokens": 8191,
        "input_cost_per_second": 0.00972,
        "output_cost_per_second": 0.00972,
        "litellm_provider": "bedrock",
        "mode": "chat"
    },
    "anthropic.claude-instant-v1": {
        "max_tokens": 8191, 
        "max_input_tokens": 100000, 
        "max_output_tokens": 8191,
        "input_cost_per_token": 0.00000163,
        "output_cost_per_token": 0.00000551,
        "litellm_provider": "bedrock",
        "mode": "chat"
    },
    "bedrock/us-east-1/anthropic.claude-instant-v1": {
        "max_tokens": 8191, 
        "max_input_tokens": 100000, 
        "max_output_tokens": 8191,
        "input_cost_per_token": 0.0000008,
        "output_cost_per_token": 0.0000024,
        "litellm_provider": "bedrock",
        "mode": "chat"
    },
    "bedrock/us-east-1/1-month-commitment/anthropic.claude-instant-v1": {
        "max_tokens": 8191, 
        "max_input_tokens": 100000, 
        "max_output_tokens": 8191,
        "input_cost_per_second": 0.011,
        "output_cost_per_second": 0.011,
        "litellm_provider": "bedrock",
        "mode": "chat"
    },
    "bedrock/us-east-1/6-month-commitment/anthropic.claude-instant-v1": {
        "max_tokens": 8191, 
        "max_input_tokens": 100000, 
        "max_output_tokens": 8191,
        "input_cost_per_second": 0.00611,
        "output_cost_per_second": 0.00611,
        "litellm_provider": "bedrock",
        "mode": "chat"
    },
    "bedrock/us-west-2/1-month-commitment/anthropic.claude-instant-v1": {
        "max_tokens": 8191, 
        "max_input_tokens": 100000, 
        "max_output_tokens": 8191,
        "input_cost_per_second": 0.011,
        "output_cost_per_second": 0.011,
        "litellm_provider": "bedrock",
        "mode": "chat"
    },
    "bedrock/us-west-2/6-month-commitment/anthropic.claude-instant-v1": {
        "max_tokens": 8191, 
        "max_input_tokens": 100000, 
        "max_output_tokens": 8191,
        "input_cost_per_second": 0.00611,
        "output_cost_per_second": 0.00611,
        "litellm_provider": "bedrock",
        "mode": "chat"
    },
    "bedrock/us-west-2/anthropic.claude-instant-v1": {
        "max_tokens": 8191, 
        "max_input_tokens": 100000, 
        "max_output_tokens": 8191,
        "input_cost_per_token": 0.0000008,
        "output_cost_per_token": 0.0000024,
        "litellm_provider": "bedrock",
        "mode": "chat"
    },
    "bedrock/ap-northeast-1/anthropic.claude-instant-v1": {
        "max_tokens": 8191, 
        "max_input_tokens": 100000, 
        "max_output_tokens": 8191,
        "input_cost_per_token": 0.00000223,
        "output_cost_per_token": 0.00000755,
        "litellm_provider": "bedrock",
        "mode": "chat"
    },
    "bedrock/ap-northeast-1/1-month-commitment/anthropic.claude-instant-v1": {
        "max_tokens": 8191, 
        "max_input_tokens": 100000, 
        "max_output_tokens": 8191,
        "input_cost_per_second": 0.01475,
        "output_cost_per_second": 0.01475,
        "litellm_provider": "bedrock",
        "mode": "chat"
    },
    "bedrock/ap-northeast-1/6-month-commitment/anthropic.claude-instant-v1": {
        "max_tokens": 8191, 
        "max_input_tokens": 100000, 
        "max_output_tokens": 8191,
        "input_cost_per_second": 0.008194,
        "output_cost_per_second": 0.008194,
        "litellm_provider": "bedrock",
        "mode": "chat"
    },
    "bedrock/eu-central-1/anthropic.claude-instant-v1": {
        "max_tokens": 8191, 
        "max_input_tokens": 100000, 
        "max_output_tokens": 8191,
        "input_cost_per_token": 0.00000248,
        "output_cost_per_token": 0.00000838,
        "litellm_provider": "bedrock",
        "mode": "chat"
    },
    "bedrock/eu-central-1/1-month-commitment/anthropic.claude-instant-v1": {
        "max_tokens": 8191, 
        "max_input_tokens": 100000, 
        "max_output_tokens": 8191,
        "input_cost_per_second": 0.01635,
        "output_cost_per_second": 0.01635,
        "litellm_provider": "bedrock",
        "mode": "chat"
    },
    "bedrock/eu-central-1/6-month-commitment/anthropic.claude-instant-v1": {
        "max_tokens": 8191, 
        "max_input_tokens": 100000, 
        "max_output_tokens": 8191,
        "input_cost_per_second": 0.009083,
        "output_cost_per_second": 0.009083,
        "litellm_provider": "bedrock",
        "mode": "chat"
    },
    "cohere.command-text-v14": {
        "max_tokens": 4096, 
        "max_input_tokens": 4096,
        "max_output_tokens": 4096, 
        "input_cost_per_token": 0.0000015,
        "output_cost_per_token": 0.0000020,
        "litellm_provider": "bedrock",
        "mode": "chat"
    },
    "bedrock/*/1-month-commitment/cohere.command-text-v14": {
        "max_tokens": 4096, 
        "max_input_tokens": 4096,
        "max_output_tokens": 4096, 
        "input_cost_per_second": 0.011,
        "output_cost_per_second": 0.011,
        "litellm_provider": "bedrock",
        "mode": "chat"
    },
    "bedrock/*/6-month-commitment/cohere.command-text-v14": {
        "max_tokens": 4096, 
        "max_input_tokens": 4096,
        "max_output_tokens": 4096, 
        "input_cost_per_second": 0.0066027,
        "output_cost_per_second": 0.0066027,
        "litellm_provider": "bedrock",
        "mode": "chat"
    },
    "cohere.command-light-text-v14": {
        "max_tokens": 4096, 
        "max_input_tokens": 4096,
        "max_output_tokens": 4096, 
        "input_cost_per_token": 0.0000003,
        "output_cost_per_token": 0.0000006,
        "litellm_provider": "bedrock",
        "mode": "chat"
    },
    "bedrock/*/1-month-commitment/cohere.command-light-text-v14": {
        "max_tokens": 4096, 
        "max_input_tokens": 4096,
        "max_output_tokens": 4096, 
        "input_cost_per_second": 0.001902,
        "output_cost_per_second": 0.001902,
        "litellm_provider": "bedrock",
        "mode": "chat"
    },
    "bedrock/*/6-month-commitment/cohere.command-light-text-v14": {
        "max_tokens": 4096, 
        "max_input_tokens": 4096,
        "max_output_tokens": 4096, 
        "input_cost_per_second": 0.0011416,
        "output_cost_per_second": 0.0011416,
        "litellm_provider": "bedrock",
        "mode": "chat"
    },
    "cohere.command-r-plus-v1:0": {
        "max_tokens": 4096, 
        "max_input_tokens": 128000,
        "max_output_tokens": 4096,
        "input_cost_per_token": 0.0000030,
        "output_cost_per_token": 0.000015,
        "litellm_provider": "bedrock",
        "mode": "chat"
    },
    "cohere.command-r-v1:0": {
        "max_tokens": 4096, 
        "max_input_tokens": 128000,
        "max_output_tokens": 4096,
        "input_cost_per_token": 0.0000005,
        "output_cost_per_token": 0.0000015,
        "litellm_provider": "bedrock",
        "mode": "chat"
    },
    "cohere.embed-english-v3": {
        "max_tokens": 512, 
        "max_input_tokens": 512, 
        "input_cost_per_token": 0.0000001,
        "output_cost_per_token": 0.000000,
        "litellm_provider": "bedrock",
        "mode": "embedding"
    },
    "cohere.embed-multilingual-v3": {
        "max_tokens": 512, 
        "max_input_tokens": 512, 
        "input_cost_per_token": 0.0000001,
        "output_cost_per_token": 0.000000,
        "litellm_provider": "bedrock",
        "mode": "embedding"
    },
    "meta.llama2-13b-chat-v1": {
        "max_tokens": 4096, 
        "max_input_tokens": 4096, 
        "max_output_tokens": 4096, 
        "input_cost_per_token": 0.00000075,
        "output_cost_per_token": 0.000001,
        "litellm_provider": "bedrock",
        "mode": "chat"
    },
    "meta.llama2-70b-chat-v1": {
        "max_tokens": 4096, 
        "max_input_tokens": 4096, 
        "max_output_tokens": 4096, 
        "input_cost_per_token": 0.00000195,
        "output_cost_per_token": 0.00000256,
        "litellm_provider": "bedrock",
        "mode": "chat"
    },
    "meta.llama3-8b-instruct-v1:0": {
        "max_tokens": 8192, 
        "max_input_tokens": 8192, 
        "max_output_tokens": 8192, 
        "input_cost_per_token": 0.0000003,
        "output_cost_per_token": 0.0000006,
        "litellm_provider": "bedrock",
        "mode": "chat"
    },
    "bedrock/us-east-1/meta.llama3-8b-instruct-v1:0": {
        "max_tokens": 8192, 
        "max_input_tokens": 8192, 
        "max_output_tokens": 8192, 
        "input_cost_per_token": 0.0000003,
        "output_cost_per_token": 0.0000006,
        "litellm_provider": "bedrock",
        "mode": "chat"
    },
    "bedrock/us-west-1/meta.llama3-8b-instruct-v1:0": {
        "max_tokens": 8192, 
        "max_input_tokens": 8192, 
        "max_output_tokens": 8192, 
        "input_cost_per_token": 0.0000003,
        "output_cost_per_token": 0.0000006,
        "litellm_provider": "bedrock",
        "mode": "chat"
    },
    "bedrock/ap-south-1/meta.llama3-8b-instruct-v1:0": {
        "max_tokens": 8192, 
        "max_input_tokens": 8192, 
        "max_output_tokens": 8192, 
        "input_cost_per_token": 0.00000036,
        "output_cost_per_token": 0.00000072,
        "litellm_provider": "bedrock",
        "mode": "chat"
    },
    "bedrock/ca-central-1/meta.llama3-8b-instruct-v1:0": {
        "max_tokens": 8192, 
        "max_input_tokens": 8192, 
        "max_output_tokens": 8192, 
        "input_cost_per_token": 0.00000035,
        "output_cost_per_token": 0.00000069,
        "litellm_provider": "bedrock",
        "mode": "chat"
    },
    "bedrock/eu-west-1/meta.llama3-8b-instruct-v1:0": {
        "max_tokens": 8192, 
        "max_input_tokens": 8192, 
        "max_output_tokens": 8192, 
        "input_cost_per_token": 0.00000032,
        "output_cost_per_token": 0.00000065,
        "litellm_provider": "bedrock",
        "mode": "chat"
    },
    "bedrock/eu-west-2/meta.llama3-8b-instruct-v1:0": {
        "max_tokens": 8192, 
        "max_input_tokens": 8192, 
        "max_output_tokens": 8192, 
        "input_cost_per_token": 0.00000039,
        "output_cost_per_token": 0.00000078,
        "litellm_provider": "bedrock",
        "mode": "chat"
    },
    "bedrock/sa-east-1/meta.llama3-8b-instruct-v1:0": {
        "max_tokens": 8192, 
        "max_input_tokens": 8192, 
        "max_output_tokens": 8192, 
        "input_cost_per_token": 0.0000005,
        "output_cost_per_token": 0.00000101,
        "litellm_provider": "bedrock",
        "mode": "chat"
    },
    "meta.llama3-70b-instruct-v1:0": {
        "max_tokens": 8192, 
        "max_input_tokens": 8192, 
        "max_output_tokens": 8192, 
        "input_cost_per_token": 0.00000265,
        "output_cost_per_token": 0.0000035,
        "litellm_provider": "bedrock",
        "mode": "chat"
    },
    "bedrock/us-east-1/meta.llama3-70b-instruct-v1:0": {
        "max_tokens": 8192, 
        "max_input_tokens": 8192, 
        "max_output_tokens": 8192, 
        "input_cost_per_token": 0.00000265,
        "output_cost_per_token": 0.0000035,
        "litellm_provider": "bedrock",
        "mode": "chat"
    },
    "bedrock/us-west-1/meta.llama3-70b-instruct-v1:0": {
        "max_tokens": 8192, 
        "max_input_tokens": 8192, 
        "max_output_tokens": 8192, 
        "input_cost_per_token": 0.00000265,
        "output_cost_per_token": 0.0000035,
        "litellm_provider": "bedrock",
        "mode": "chat"
    },
    "bedrock/ap-south-1/meta.llama3-70b-instruct-v1:0": {
        "max_tokens": 8192, 
        "max_input_tokens": 8192, 
        "max_output_tokens": 8192, 
        "input_cost_per_token": 0.00000318,
        "output_cost_per_token": 0.0000042,
        "litellm_provider": "bedrock",
        "mode": "chat"
    },
    "bedrock/ca-central-1/meta.llama3-70b-instruct-v1:0": {
        "max_tokens": 8192, 
        "max_input_tokens": 8192, 
        "max_output_tokens": 8192, 
        "input_cost_per_token": 0.00000305,
        "output_cost_per_token": 0.00000403,
        "litellm_provider": "bedrock",
        "mode": "chat"
    },
    "bedrock/eu-west-1/meta.llama3-70b-instruct-v1:0": {
        "max_tokens": 8192, 
        "max_input_tokens": 8192, 
        "max_output_tokens": 8192, 
        "input_cost_per_token": 0.00000286,
        "output_cost_per_token": 0.00000378,
        "litellm_provider": "bedrock",
        "mode": "chat"
    },
    "bedrock/eu-west-2/meta.llama3-70b-instruct-v1:0": {
        "max_tokens": 8192, 
        "max_input_tokens": 8192, 
        "max_output_tokens": 8192, 
        "input_cost_per_token": 0.00000345,
        "output_cost_per_token": 0.00000455,
        "litellm_provider": "bedrock",
        "mode": "chat"
    },
    "bedrock/sa-east-1/meta.llama3-70b-instruct-v1:0": {
        "max_tokens": 8192, 
        "max_input_tokens": 8192, 
        "max_output_tokens": 8192, 
        "input_cost_per_token": 0.00000445,
        "output_cost_per_token": 0.00000588,
        "litellm_provider": "bedrock",
        "mode": "chat"
    },
    "meta.llama3-1-8b-instruct-v1:0": {
        "max_tokens": 128000,
        "max_input_tokens": 128000,
        "max_output_tokens": 2048,
        "input_cost_per_token": 0.00000022,
        "output_cost_per_token": 0.00000022,
        "litellm_provider": "bedrock",
        "mode": "chat",
        "supports_function_calling": true, 
        "supports_tool_choice": false
    },
    "meta.llama3-1-70b-instruct-v1:0": {
        "max_tokens": 128000,
        "max_input_tokens": 128000,
        "max_output_tokens": 2048,
        "input_cost_per_token": 0.00000099,
        "output_cost_per_token": 0.00000099,
        "litellm_provider": "bedrock",
        "mode": "chat",
        "supports_function_calling": true, 
        "supports_tool_choice": false
    },
    "meta.llama3-1-405b-instruct-v1:0": {
        "max_tokens": 128000,
        "max_input_tokens": 128000,
        "max_output_tokens": 4096,
        "input_cost_per_token": 0.00000532,
        "output_cost_per_token": 0.000016,
        "litellm_provider": "bedrock",
        "mode": "chat",
        "supports_function_calling": true, 
        "supports_tool_choice": false
    },
    "512-x-512/50-steps/stability.stable-diffusion-xl-v0": {
        "max_tokens": 77, 
        "max_input_tokens": 77, 
        "output_cost_per_image": 0.018,
        "litellm_provider": "bedrock",
        "mode": "image_generation"
    },
    "512-x-512/max-steps/stability.stable-diffusion-xl-v0": {
        "max_tokens": 77, 
        "max_input_tokens": 77, 
        "output_cost_per_image": 0.036,
        "litellm_provider": "bedrock",
        "mode": "image_generation"
    },
    "max-x-max/50-steps/stability.stable-diffusion-xl-v0": {
        "max_tokens": 77, 
        "max_input_tokens": 77, 
        "output_cost_per_image": 0.036,
        "litellm_provider": "bedrock",
        "mode": "image_generation"
    },
    "max-x-max/max-steps/stability.stable-diffusion-xl-v0": {
        "max_tokens": 77, 
        "max_input_tokens": 77, 
        "output_cost_per_image": 0.072,
        "litellm_provider": "bedrock",
        "mode": "image_generation"
    },
    "1024-x-1024/50-steps/stability.stable-diffusion-xl-v1": {
        "max_tokens": 77, 
        "max_input_tokens": 77, 
        "output_cost_per_image": 0.04,
        "litellm_provider": "bedrock",
        "mode": "image_generation"
    },
    "1024-x-1024/max-steps/stability.stable-diffusion-xl-v1": {
        "max_tokens": 77, 
        "max_input_tokens": 77, 
        "output_cost_per_image": 0.08,
        "litellm_provider": "bedrock",
        "mode": "image_generation"
    },
    "sagemaker/meta-textgeneration-llama-2-7b": {
        "max_tokens": 4096, 
        "max_input_tokens": 4096, 
        "max_output_tokens": 4096, 
        "input_cost_per_token": 0.000,
        "output_cost_per_token": 0.000,
        "litellm_provider": "sagemaker",
        "mode": "completion"
    },
    "sagemaker/meta-textgeneration-llama-2-7b-f": {
        "max_tokens": 4096, 
        "max_input_tokens": 4096, 
        "max_output_tokens": 4096, 
        "input_cost_per_token": 0.000,
        "output_cost_per_token": 0.000,
        "litellm_provider": "sagemaker",
        "mode": "chat"
    },
    "sagemaker/meta-textgeneration-llama-2-13b": {
        "max_tokens": 4096, 
        "max_input_tokens": 4096, 
        "max_output_tokens": 4096, 
        "input_cost_per_token": 0.000,
        "output_cost_per_token": 0.000,
        "litellm_provider": "sagemaker",
        "mode": "completion"
    },
    "sagemaker/meta-textgeneration-llama-2-13b-f": {
        "max_tokens": 4096, 
        "max_input_tokens": 4096, 
        "max_output_tokens": 4096, 
        "input_cost_per_token": 0.000,
        "output_cost_per_token": 0.000,
        "litellm_provider": "sagemaker",
        "mode": "chat"
    },
    "sagemaker/meta-textgeneration-llama-2-70b": {
        "max_tokens": 4096, 
        "max_input_tokens": 4096, 
        "max_output_tokens": 4096, 
        "input_cost_per_token": 0.000,
        "output_cost_per_token": 0.000,
        "litellm_provider": "sagemaker",
        "mode": "completion"
    },
    "sagemaker/meta-textgeneration-llama-2-70b-b-f": {
        "max_tokens": 4096, 
        "max_input_tokens": 4096, 
        "max_output_tokens": 4096, 
        "input_cost_per_token": 0.000,
        "output_cost_per_token": 0.000,
        "litellm_provider": "sagemaker",
        "mode": "chat"
    },
    "together-ai-up-to-4b": {
        "input_cost_per_token": 0.0000001,
        "output_cost_per_token": 0.0000001,
        "litellm_provider": "together_ai"
    },
    "together-ai-4.1b-8b": {
        "input_cost_per_token": 0.0000002,
        "output_cost_per_token": 0.0000002,
        "litellm_provider": "together_ai"
    },
    "together-ai-8.1b-21b": {
        "max_tokens": 1000,
        "input_cost_per_token": 0.0000003,
        "output_cost_per_token": 0.0000003,
        "litellm_provider": "together_ai"
    },
    "together-ai-21.1b-41b": {
        "input_cost_per_token": 0.0000008,
        "output_cost_per_token": 0.0000008,
        "litellm_provider": "together_ai"
    },
    "together-ai-41.1b-80b": {
        "input_cost_per_token": 0.0000009,
        "output_cost_per_token": 0.0000009,
        "litellm_provider": "together_ai"
    },
    "together-ai-81.1b-110b": {
        "input_cost_per_token": 0.0000018,
        "output_cost_per_token": 0.0000018,
        "litellm_provider": "together_ai"
    },
    "together_ai/mistralai/Mixtral-8x7B-Instruct-v0.1": {
        "input_cost_per_token": 0.0000006,
        "output_cost_per_token": 0.0000006,
        "litellm_provider": "together_ai",
        "supports_function_calling": true,
        "supports_parallel_function_calling": true
    },
    "together_ai/mistralai/Mistral-7B-Instruct-v0.1": {
        "litellm_provider": "together_ai",
        "supports_function_calling": true,
        "supports_parallel_function_calling": true
    },
    "together_ai/togethercomputer/CodeLlama-34b-Instruct": {
        "litellm_provider": "together_ai",
        "supports_function_calling": true,
        "supports_parallel_function_calling": true
    },
    "ollama/codegemma": {
        "max_tokens": 8192, 
        "max_input_tokens": 8192, 
        "max_output_tokens": 8192, 
        "input_cost_per_token": 0.0,
        "output_cost_per_token": 0.0,
        "litellm_provider": "ollama",
        "mode": "completion"
    },
    "ollama/codegeex4": {
        "max_tokens": 32768,
        "max_input_tokens": 32768,
        "max_output_tokens": 8192,
        "input_cost_per_token": 0.0,
        "output_cost_per_token": 0.0,
        "litellm_provider": "ollama",
        "mode": "chat", 
        "supports_function_calling": false
    },
    "ollama/deepseek-coder-v2-instruct": {
        "max_tokens": 32768,
        "max_input_tokens": 32768,
        "max_output_tokens": 8192,
        "input_cost_per_token": 0.0,
        "output_cost_per_token": 0.0,
        "litellm_provider": "ollama",
        "mode": "chat", 
        "supports_function_calling": true
    },
    "ollama/deepseek-coder-v2-base": {
        "max_tokens": 8192,
        "max_input_tokens": 8192,
        "max_output_tokens": 8192,
        "input_cost_per_token": 0.0,
        "output_cost_per_token": 0.0,
        "litellm_provider": "ollama",
        "mode": "completion", 
        "supports_function_calling": true
    },
    "ollama/deepseek-coder-v2-lite-instruct": {
        "max_tokens": 32768,
        "max_input_tokens": 32768,
        "max_output_tokens": 8192,
        "input_cost_per_token": 0.0,
        "output_cost_per_token": 0.0,
        "litellm_provider": "ollama",
        "mode": "chat", 
        "supports_function_calling": true
    },
    "ollama/deepseek-coder-v2-lite-base": {
        "max_tokens": 8192,
        "max_input_tokens": 8192,
        "max_output_tokens": 8192,
        "input_cost_per_token": 0.0,
        "output_cost_per_token": 0.0,
        "litellm_provider": "ollama",
        "mode": "completion", 
        "supports_function_calling": true
    },
    "ollama/internlm2_5-20b-chat": {
        "max_tokens": 32768,
        "max_input_tokens": 32768,
        "max_output_tokens": 8192,
        "input_cost_per_token": 0.0,
        "output_cost_per_token": 0.0,
        "litellm_provider": "ollama",
        "mode": "chat", 
        "supports_function_calling": true
    },
    "ollama/llama2": {
        "max_tokens": 4096, 
        "max_input_tokens": 4096, 
        "max_output_tokens": 4096, 
        "input_cost_per_token": 0.0,
        "output_cost_per_token": 0.0,
        "litellm_provider": "ollama",
        "mode": "completion"
    },
    "ollama/llama2:7b": {
        "max_tokens": 4096, 
        "max_input_tokens": 4096, 
        "max_output_tokens": 4096, 
        "input_cost_per_token": 0.0,
        "output_cost_per_token": 0.0,
        "litellm_provider": "ollama",
        "mode": "completion"
    },
    "ollama/llama2:13b": {
        "max_tokens": 4096, 
        "max_input_tokens": 4096, 
        "max_output_tokens": 4096, 
        "input_cost_per_token": 0.0,
        "output_cost_per_token": 0.0,
        "litellm_provider": "ollama",
        "mode": "completion"
    },
    "ollama/llama2:70b": {
        "max_tokens": 4096, 
        "max_input_tokens": 4096, 
        "max_output_tokens": 4096, 
        "input_cost_per_token": 0.0,
        "output_cost_per_token": 0.0,
        "litellm_provider": "ollama",
        "mode": "completion"
    },
    "ollama/llama2-uncensored": {
        "max_tokens": 4096, 
        "max_input_tokens": 4096, 
        "max_output_tokens": 4096, 
        "input_cost_per_token": 0.0,
        "output_cost_per_token": 0.0,
        "litellm_provider": "ollama",
        "mode": "completion"
    },
    "ollama/llama3": {
        "max_tokens": 8192,
        "max_input_tokens": 8192,
        "max_output_tokens": 8192,
        "input_cost_per_token": 0.0,
        "output_cost_per_token": 0.0,
        "litellm_provider": "ollama",
        "mode": "chat"
    },
    "ollama/llama3:8b": {
        "max_tokens": 8192,
        "max_input_tokens": 8192,
        "max_output_tokens": 8192,
        "input_cost_per_token": 0.0,
        "output_cost_per_token": 0.0,
        "litellm_provider": "ollama",
        "mode": "chat"
    },
    "ollama/llama3:70b": {
        "max_tokens": 8192,
        "max_input_tokens": 8192,
        "max_output_tokens": 8192,
        "input_cost_per_token": 0.0,
        "output_cost_per_token": 0.0,
        "litellm_provider": "ollama",
        "mode": "chat"
    },
    "ollama/llama3.1": {
        "max_tokens": 32768,
        "max_input_tokens": 8192,
        "max_output_tokens": 8192,
        "input_cost_per_token": 0.0,
        "output_cost_per_token": 0.0,
        "litellm_provider": "ollama",
        "mode": "chat", 
        "supports_function_calling": true
    },
    "ollama/mistral-large-instruct-2407": {
        "max_tokens": 65536,
        "max_input_tokens": 65536,
        "max_output_tokens": 8192,
        "input_cost_per_token": 0.0,
        "output_cost_per_token": 0.0,
        "litellm_provider": "ollama",
        "mode": "chat"
    },
    "ollama/mistral": {
        "max_tokens": 8192,
        "max_input_tokens": 8192,
        "max_output_tokens": 8192,
        "input_cost_per_token": 0.0,
        "output_cost_per_token": 0.0,
        "litellm_provider": "ollama",
        "mode": "completion"
    },
    "ollama/mistral-7B-Instruct-v0.1": {
        "max_tokens": 8192,
        "max_input_tokens": 8192,
        "max_output_tokens": 8192,
        "input_cost_per_token": 0.0,
        "output_cost_per_token": 0.0,
        "litellm_provider": "ollama",
        "mode": "chat"
    },
    "ollama/mistral-7B-Instruct-v0.2": {
        "max_tokens": 32768,
        "max_input_tokens": 32768,
        "max_output_tokens": 32768,
        "input_cost_per_token": 0.0,
        "output_cost_per_token": 0.0,
        "litellm_provider": "ollama",
        "mode": "chat"
    },
    "ollama/mixtral-8x7B-Instruct-v0.1": {
        "max_tokens": 32768,
        "max_input_tokens": 32768,
        "max_output_tokens": 32768,
        "input_cost_per_token": 0.0,
        "output_cost_per_token": 0.0,
        "litellm_provider": "ollama",
        "mode": "chat"
    },
    "ollama/mixtral-8x22B-Instruct-v0.1": {
        "max_tokens": 65536,
        "max_input_tokens": 65536,
        "max_output_tokens": 65536,
        "input_cost_per_token": 0.0,
        "output_cost_per_token": 0.0,
        "litellm_provider": "ollama",
        "mode": "chat"
    },
    "ollama/codellama": {
        "max_tokens": 4096, 
        "max_input_tokens": 4096, 
        "max_output_tokens": 4096,
        "input_cost_per_token": 0.0,
        "output_cost_per_token": 0.0,
        "litellm_provider": "ollama",
        "mode": "completion"
    },
    "ollama/orca-mini": {
        "max_tokens": 4096, 
        "max_input_tokens": 4096, 
        "max_output_tokens": 4096,
        "input_cost_per_token": 0.0,
        "output_cost_per_token": 0.0,
        "litellm_provider": "ollama",
        "mode": "completion"
    },
    "ollama/vicuna": {
        "max_tokens": 2048,
        "max_input_tokens": 2048,
        "max_output_tokens": 2048,
        "input_cost_per_token": 0.0,
        "output_cost_per_token": 0.0,
        "litellm_provider": "ollama",
        "mode": "completion"
    },
    "deepinfra/lizpreciatior/lzlv_70b_fp16_hf": {
        "max_tokens": 4096,
        "max_input_tokens": 4096,
        "max_output_tokens": 4096,
        "input_cost_per_token": 0.00000070,
        "output_cost_per_token": 0.00000090,
        "litellm_provider": "deepinfra",
        "mode": "chat"
    },
    "deepinfra/Gryphe/MythoMax-L2-13b": {
        "max_tokens": 4096,
        "max_input_tokens": 4096,
        "max_output_tokens": 4096,
        "input_cost_per_token": 0.00000022,
        "output_cost_per_token": 0.00000022,
        "litellm_provider": "deepinfra",
        "mode": "chat"
    },
    "deepinfra/mistralai/Mistral-7B-Instruct-v0.1": {
        "max_tokens": 8191,
        "max_input_tokens": 32768,
        "max_output_tokens": 8191,
        "input_cost_per_token": 0.00000013,
        "output_cost_per_token": 0.00000013,
        "litellm_provider": "deepinfra",
        "mode": "chat"
    },
    "deepinfra/meta-llama/Llama-2-70b-chat-hf": {
        "max_tokens": 4096,
        "max_input_tokens": 4096,
        "max_output_tokens": 4096,
        "input_cost_per_token": 0.00000070,
        "output_cost_per_token": 0.00000090,
        "litellm_provider": "deepinfra",
        "mode": "chat"
    },
    "deepinfra/cognitivecomputations/dolphin-2.6-mixtral-8x7b": {
        "max_tokens": 8191,
        "max_input_tokens": 32768,
        "max_output_tokens": 8191,
        "input_cost_per_token": 0.00000027,
        "output_cost_per_token": 0.00000027,
        "litellm_provider": "deepinfra",
        "mode": "chat"
    },
    "deepinfra/codellama/CodeLlama-34b-Instruct-hf": {
        "max_tokens": 4096,
        "max_input_tokens": 4096,
        "max_output_tokens": 4096,
        "input_cost_per_token": 0.00000060,
        "output_cost_per_token": 0.00000060,
        "litellm_provider": "deepinfra",
        "mode": "chat"
    },
    "deepinfra/deepinfra/mixtral": {
        "max_tokens": 4096,
        "max_input_tokens": 32000,
        "max_output_tokens": 4096,
        "input_cost_per_token": 0.00000027,
        "output_cost_per_token": 0.00000027,
        "litellm_provider": "deepinfra",
        "mode": "completion"
    },
    "deepinfra/Phind/Phind-CodeLlama-34B-v2": {
        "max_tokens": 4096,
        "max_input_tokens": 16384,
        "max_output_tokens": 4096,
        "input_cost_per_token": 0.00000060,
        "output_cost_per_token": 0.00000060,
        "litellm_provider": "deepinfra",
        "mode": "chat"
    },
    "deepinfra/mistralai/Mixtral-8x7B-Instruct-v0.1": {
        "max_tokens": 8191,
        "max_input_tokens": 32768,
        "max_output_tokens": 8191,
        "input_cost_per_token": 0.00000027,
        "output_cost_per_token": 0.00000027,
        "litellm_provider": "deepinfra",
        "mode": "chat"
    },
    "deepinfra/deepinfra/airoboros-70b": {
        "max_tokens": 4096,
        "max_input_tokens": 4096,
        "max_output_tokens": 4096,
        "input_cost_per_token": 0.00000070,
        "output_cost_per_token": 0.00000090,
        "litellm_provider": "deepinfra",
        "mode": "chat"
    },
    "deepinfra/01-ai/Yi-34B-Chat": {
        "max_tokens": 4096,
        "max_input_tokens": 4096,
        "max_output_tokens": 4096,
        "input_cost_per_token": 0.00000060,
        "output_cost_per_token": 0.00000060,
        "litellm_provider": "deepinfra",
        "mode": "chat"
    },
    "deepinfra/01-ai/Yi-6B-200K": {
        "max_tokens": 4096,
        "max_input_tokens": 200000,
        "max_output_tokens": 4096,
        "input_cost_per_token": 0.00000013,
        "output_cost_per_token": 0.00000013,
        "litellm_provider": "deepinfra",
        "mode": "completion"
    },
    "deepinfra/jondurbin/airoboros-l2-70b-gpt4-1.4.1": {
        "max_tokens": 4096,
        "max_input_tokens": 4096,
        "max_output_tokens": 4096,
        "input_cost_per_token": 0.00000070,
        "output_cost_per_token": 0.00000090,
        "litellm_provider": "deepinfra",
        "mode": "chat"
    },
    "deepinfra/meta-llama/Llama-2-13b-chat-hf": {
        "max_tokens": 4096,
        "max_input_tokens": 4096,
        "max_output_tokens": 4096,
        "input_cost_per_token": 0.00000022,
        "output_cost_per_token": 0.00000022,
        "litellm_provider": "deepinfra",
        "mode": "chat"
    },
    "deepinfra/amazon/MistralLite": {
        "max_tokens": 8191,
        "max_input_tokens": 32768,
        "max_output_tokens": 8191,
        "input_cost_per_token": 0.00000020,
        "output_cost_per_token": 0.00000020,
        "litellm_provider": "deepinfra",
        "mode": "chat"
    },
    "deepinfra/meta-llama/Llama-2-7b-chat-hf": {
        "max_tokens": 4096,
        "max_input_tokens": 4096,
        "max_output_tokens": 4096,
        "input_cost_per_token": 0.00000013,
        "output_cost_per_token": 0.00000013,
        "litellm_provider": "deepinfra",
        "mode": "chat"
    },
    "deepinfra/meta-llama/Meta-Llama-3-8B-Instruct": {
        "max_tokens": 8191,
        "max_input_tokens": 8191,
        "max_output_tokens": 4096,
        "input_cost_per_token": 0.00000008,
        "output_cost_per_token": 0.00000008,
        "litellm_provider": "deepinfra",
        "mode": "chat"
    },
    "deepinfra/meta-llama/Meta-Llama-3-70B-Instruct": {
        "max_tokens": 8191,
        "max_input_tokens": 8191,
        "max_output_tokens": 4096,
        "input_cost_per_token": 0.00000059,
        "output_cost_per_token": 0.00000079,
        "litellm_provider": "deepinfra",
        "mode": "chat"
    },
    "deepinfra/01-ai/Yi-34B-200K": {
        "max_tokens": 4096,
        "max_input_tokens": 200000,
        "max_output_tokens": 4096,
        "input_cost_per_token": 0.00000060,
        "output_cost_per_token": 0.00000060,
        "litellm_provider": "deepinfra",
        "mode": "completion"
    },
    "deepinfra/openchat/openchat_3.5": {
        "max_tokens": 4096,
        "max_input_tokens": 4096,
        "max_output_tokens": 4096,
        "input_cost_per_token": 0.00000013,
        "output_cost_per_token": 0.00000013,
        "litellm_provider": "deepinfra",
        "mode": "chat"
    },
    "perplexity/codellama-34b-instruct": { 
        "max_tokens": 16384,
        "max_input_tokens": 16384,
        "max_output_tokens": 16384,
        "input_cost_per_token": 0.00000035, 
        "output_cost_per_token": 0.00000140,  
        "litellm_provider": "perplexity", 
        "mode": "chat" 
    },
    "perplexity/codellama-70b-instruct": { 
        "max_tokens": 16384,
        "max_input_tokens": 16384,
        "max_output_tokens": 16384,
        "input_cost_per_token": 0.00000070, 
        "output_cost_per_token": 0.00000280,  
        "litellm_provider": "perplexity", 
        "mode": "chat" 
    },
    "perplexity/llama-3.1-70b-instruct": { 
        "max_tokens": 131072,
        "max_input_tokens": 131072,
        "max_output_tokens": 131072,
        "input_cost_per_token": 0.000001, 
        "output_cost_per_token": 0.000001,
        "litellm_provider": "perplexity", 
        "mode": "chat" 
    },
    "perplexity/llama-3.1-8b-instruct": { 
        "max_tokens": 131072,
        "max_input_tokens": 131072,
        "max_output_tokens": 131072,
        "input_cost_per_token": 0.0000002, 
        "output_cost_per_token": 0.0000002,  
        "litellm_provider": "perplexity", 
        "mode": "chat" 
    },
    "perplexity/llama-3.1-sonar-huge-128k-online": { 
        "max_tokens": 127072,
        "max_input_tokens": 127072,
        "max_output_tokens": 127072,
        "input_cost_per_token": 0.000005, 
        "output_cost_per_token": 0.000005,
        "litellm_provider": "perplexity", 
        "mode": "chat" 
    },
    "perplexity/llama-3.1-sonar-large-128k-online": { 
        "max_tokens": 127072,
        "max_input_tokens": 127072,
        "max_output_tokens": 127072,
        "input_cost_per_token": 0.000001, 
        "output_cost_per_token": 0.000001,
        "litellm_provider": "perplexity", 
        "mode": "chat" 
    },
    "perplexity/llama-3.1-sonar-large-128k-chat": { 
        "max_tokens": 131072,
        "max_input_tokens": 131072,
        "max_output_tokens": 131072,
        "input_cost_per_token": 0.000001, 
        "output_cost_per_token": 0.000001,
        "litellm_provider": "perplexity", 
        "mode": "chat" 
    },
    "perplexity/llama-3.1-sonar-small-128k-chat": { 
        "max_tokens": 131072,
        "max_input_tokens": 131072,
        "max_output_tokens": 131072,
        "input_cost_per_token": 0.0000002, 
        "output_cost_per_token": 0.0000002,  
        "litellm_provider": "perplexity", 
        "mode": "chat" 
    },
    "perplexity/llama-3.1-sonar-small-128k-online": { 
        "max_tokens": 127072,
        "max_input_tokens": 127072,
        "max_output_tokens": 127072,
        "input_cost_per_token": 0.0000002, 
        "output_cost_per_token": 0.0000002,  
        "litellm_provider": "perplexity", 
        "mode": "chat" 
    },
    "perplexity/pplx-7b-chat": { 
        "max_tokens": 8192,
        "max_input_tokens": 8192,
        "max_output_tokens": 8192,
        "input_cost_per_token": 0.00000007, 
        "output_cost_per_token": 0.00000028, 
        "litellm_provider": "perplexity", 
        "mode": "chat" 
    },
    "perplexity/pplx-70b-chat": {  
        "max_tokens": 4096,
        "max_input_tokens": 4096,
        "max_output_tokens": 4096,
        "input_cost_per_token": 0.00000070, 
        "output_cost_per_token": 0.00000280, 
        "litellm_provider": "perplexity", 
        "mode": "chat" 
    },
    "perplexity/pplx-7b-online": { 
        "max_tokens": 4096,
        "max_input_tokens": 4096,
        "max_output_tokens": 4096,
        "input_cost_per_token": 0.0000000, 
        "output_cost_per_token": 0.00000028, 
        "input_cost_per_request": 0.005,
        "litellm_provider": "perplexity", 
        "mode": "chat" 
    },
    "perplexity/pplx-70b-online": { 
        "max_tokens": 4096,
        "max_input_tokens": 4096,
        "max_output_tokens": 4096, 
        "input_cost_per_token": 0.0000000, 
        "output_cost_per_token": 0.00000280, 
        "input_cost_per_request": 0.005,
        "litellm_provider": "perplexity", 
        "mode": "chat" 
    },
    "perplexity/llama-2-70b-chat": { 
        "max_tokens": 4096,
        "max_input_tokens": 4096,
        "max_output_tokens": 4096, 
        "input_cost_per_token": 0.00000070, 
        "output_cost_per_token": 0.00000280,
        "litellm_provider": "perplexity", 
        "mode": "chat" 
    },
    "perplexity/mistral-7b-instruct": { 
        "max_tokens": 4096,
        "max_input_tokens": 4096,
        "max_output_tokens": 4096, 
        "input_cost_per_token": 0.00000007,
        "output_cost_per_token": 0.00000028,
        "litellm_provider": "perplexity", 
        "mode": "chat" 
    },
    "perplexity/mixtral-8x7b-instruct": {
        "max_tokens": 4096,
        "max_input_tokens": 4096,
        "max_output_tokens": 4096,
        "input_cost_per_token": 0.00000007,
        "output_cost_per_token": 0.00000028,
        "litellm_provider": "perplexity",
        "mode": "chat"
    },
    "perplexity/sonar-small-chat": {
        "max_tokens": 16384,
        "max_input_tokens": 16384,
        "max_output_tokens": 16384,
        "input_cost_per_token": 0.00000007,
        "output_cost_per_token": 0.00000028,
        "litellm_provider": "perplexity",
        "mode": "chat"
    },
    "perplexity/sonar-small-online": {
        "max_tokens": 12000,
        "max_input_tokens": 12000,
        "max_output_tokens": 12000,
        "input_cost_per_token": 0,
        "output_cost_per_token": 0.00000028,
        "input_cost_per_request": 0.005,
        "litellm_provider": "perplexity",
        "mode": "chat"
    },
    "perplexity/sonar-medium-chat": {
        "max_tokens": 16384,
        "max_input_tokens": 16384,
        "max_output_tokens": 16384,
        "input_cost_per_token": 0.0000006,
        "output_cost_per_token": 0.0000018,
        "litellm_provider": "perplexity",
        "mode": "chat"
    },
    "perplexity/sonar-medium-online": {
        "max_tokens": 12000,
        "max_input_tokens": 12000,
        "max_output_tokens": 12000,
        "input_cost_per_token": 0,
        "output_cost_per_token": 0.0000018,
        "input_cost_per_request": 0.005,
        "litellm_provider": "perplexity",
        "mode": "chat"
    },
    "fireworks_ai/accounts/fireworks/models/firefunction-v2": {
        "max_tokens": 8192,
        "max_input_tokens": 8192,
        "max_output_tokens": 8192,
        "input_cost_per_token": 0.0000009, 
        "output_cost_per_token": 0.0000009,
        "litellm_provider": "fireworks_ai", 
        "mode": "chat",
        "supports_function_calling": true,
        "source": "https://fireworks.ai/pricing"
    },
    "fireworks_ai/accounts/fireworks/models/mixtral-8x22b-instruct-hf": {
        "max_tokens": 65536,
        "max_input_tokens": 65536,
        "max_output_tokens": 65536,
        "input_cost_per_token": 0.0000012, 
        "output_cost_per_token": 0.0000012,
        "litellm_provider": "fireworks_ai", 
        "mode": "chat",
        "supports_function_calling": true,
        "source": "https://fireworks.ai/pricing"
    },
    "fireworks_ai/accounts/fireworks/models/qwen2-72b-instruct": {
        "max_tokens": 32768,
        "max_input_tokens": 32768,
        "max_output_tokens": 32768,
        "input_cost_per_token": 0.0000009, 
        "output_cost_per_token": 0.0000009,
        "litellm_provider": "fireworks_ai", 
        "mode": "chat",
        "supports_function_calling": true,
        "source": "https://fireworks.ai/pricing"
    },
    "fireworks_ai/accounts/fireworks/models/yi-large": {
        "max_tokens": 32768,
        "max_input_tokens": 32768,
        "max_output_tokens": 32768,
        "input_cost_per_token": 0.000003, 
        "output_cost_per_token": 0.000003,
        "litellm_provider": "fireworks_ai", 
        "mode": "chat",
        "supports_function_calling": true,
        "source": "https://fireworks.ai/pricing"
    },
    "fireworks_ai/accounts/fireworks/models/deepseek-coder-v2-instruct": {
        "max_tokens": 65536,
        "max_input_tokens": 65536,
        "max_output_tokens": 8192,
        "input_cost_per_token": 0.0000012, 
        "output_cost_per_token": 0.0000012,
        "litellm_provider": "fireworks_ai", 
        "mode": "chat",
        "supports_function_calling": true,
        "source": "https://fireworks.ai/pricing"
    },
      "anyscale/mistralai/Mistral-7B-Instruct-v0.1": {
        "max_tokens": 16384,
        "max_input_tokens": 16384,
        "max_output_tokens": 16384,
        "input_cost_per_token": 0.00000015, 
        "output_cost_per_token": 0.00000015,
        "litellm_provider": "anyscale", 
        "mode": "chat",
        "supports_function_calling": true,
        "source": "https://docs.anyscale.com/preview/endpoints/text-generation/supported-models/mistralai-Mistral-7B-Instruct-v0.1"
      },
      "anyscale/mistralai/Mixtral-8x7B-Instruct-v0.1": {
        "max_tokens": 16384,
        "max_input_tokens": 16384,
        "max_output_tokens": 16384,
        "input_cost_per_token": 0.00000015, 
        "output_cost_per_token": 0.00000015,
        "litellm_provider": "anyscale", 
        "mode": "chat",
        "supports_function_calling": true,
        "source": "https://docs.anyscale.com/preview/endpoints/text-generation/supported-models/mistralai-Mixtral-8x7B-Instruct-v0.1"
      },
      "anyscale/mistralai/Mixtral-8x22B-Instruct-v0.1": {
        "max_tokens": 65536,
        "max_input_tokens": 65536,
        "max_output_tokens": 65536,
        "input_cost_per_token": 0.00000090, 
        "output_cost_per_token": 0.00000090,
        "litellm_provider": "anyscale", 
        "mode": "chat",
        "supports_function_calling": true,
        "source": "https://docs.anyscale.com/preview/endpoints/text-generation/supported-models/mistralai-Mixtral-8x22B-Instruct-v0.1"
      },
      "anyscale/HuggingFaceH4/zephyr-7b-beta": {
        "max_tokens": 16384,
        "max_input_tokens": 16384,
        "max_output_tokens": 16384,
        "input_cost_per_token": 0.00000015, 
        "output_cost_per_token": 0.00000015,
        "litellm_provider": "anyscale", 
        "mode": "chat"
      },
      "anyscale/google/gemma-7b-it": {
        "max_tokens": 8192,
        "max_input_tokens": 8192,
        "max_output_tokens": 8192,
        "input_cost_per_token": 0.00000015, 
        "output_cost_per_token": 0.00000015,
        "litellm_provider": "anyscale", 
        "mode": "chat",
        "source": "https://docs.anyscale.com/preview/endpoints/text-generation/supported-models/google-gemma-7b-it"
      },
      "anyscale/meta-llama/Llama-2-7b-chat-hf": {
        "max_tokens": 4096,
        "max_input_tokens": 4096,
        "max_output_tokens": 4096,
        "input_cost_per_token": 0.00000015, 
        "output_cost_per_token": 0.00000015, 
        "litellm_provider": "anyscale", 
        "mode": "chat"
      },
      "anyscale/meta-llama/Llama-2-13b-chat-hf": {
        "max_tokens": 4096,
        "max_input_tokens": 4096,
        "max_output_tokens": 4096,
        "input_cost_per_token": 0.00000025, 
        "output_cost_per_token": 0.00000025, 
        "litellm_provider": "anyscale", 
        "mode": "chat"
      },
      "anyscale/meta-llama/Llama-2-70b-chat-hf": {
        "max_tokens": 4096,
        "max_input_tokens": 4096,
        "max_output_tokens": 4096,
        "input_cost_per_token": 0.000001, 
        "output_cost_per_token": 0.000001, 
        "litellm_provider": "anyscale", 
        "mode": "chat"
      },
      "anyscale/codellama/CodeLlama-34b-Instruct-hf": {
        "max_tokens": 4096,
        "max_input_tokens": 4096,
        "max_output_tokens": 4096,
        "input_cost_per_token": 0.000001, 
        "output_cost_per_token": 0.000001, 
        "litellm_provider": "anyscale", 
        "mode": "chat"
      },
      "anyscale/codellama/CodeLlama-70b-Instruct-hf": {
        "max_tokens": 4096,
        "max_input_tokens": 4096,
        "max_output_tokens": 4096,
        "input_cost_per_token": 0.000001, 
        "output_cost_per_token": 0.000001, 
        "litellm_provider": "anyscale", 
        "mode": "chat",
        "source" : "https://docs.anyscale.com/preview/endpoints/text-generation/supported-models/codellama-CodeLlama-70b-Instruct-hf"
      },
      "anyscale/meta-llama/Meta-Llama-3-8B-Instruct": {
        "max_tokens": 8192,
        "max_input_tokens": 8192,
        "max_output_tokens": 8192,
        "input_cost_per_token": 0.00000015, 
        "output_cost_per_token": 0.00000015, 
        "litellm_provider": "anyscale", 
        "mode": "chat",
        "source": "https://docs.anyscale.com/preview/endpoints/text-generation/supported-models/meta-llama-Meta-Llama-3-8B-Instruct"
      },
      "anyscale/meta-llama/Meta-Llama-3-70B-Instruct": {
        "max_tokens": 8192,
        "max_input_tokens": 8192,
        "max_output_tokens": 8192,
        "input_cost_per_token": 0.00000100, 
        "output_cost_per_token": 0.00000100, 
        "litellm_provider": "anyscale", 
        "mode": "chat",
        "source" : "https://docs.anyscale.com/preview/endpoints/text-generation/supported-models/meta-llama-Meta-Llama-3-70B-Instruct"
      },
      "cloudflare/@cf/meta/llama-2-7b-chat-fp16": {
        "max_tokens": 3072, 
        "max_input_tokens": 3072, 
        "max_output_tokens": 3072, 
        "input_cost_per_token": 0.000001923, 
        "output_cost_per_token": 0.000001923, 
        "litellm_provider": "cloudflare", 
        "mode": "chat"
      },
      "cloudflare/@cf/meta/llama-2-7b-chat-int8": {
        "max_tokens": 2048, 
        "max_input_tokens": 2048, 
        "max_output_tokens": 2048, 
        "input_cost_per_token": 0.000001923, 
        "output_cost_per_token": 0.000001923, 
        "litellm_provider": "cloudflare", 
        "mode": "chat"
      },
      "cloudflare/@cf/mistral/mistral-7b-instruct-v0.1": {
        "max_tokens": 8192, 
        "max_input_tokens": 8192, 
        "max_output_tokens": 8192, 
        "input_cost_per_token": 0.000001923, 
        "output_cost_per_token": 0.000001923, 
        "litellm_provider": "cloudflare", 
        "mode": "chat"
      },
      "cloudflare/@hf/thebloke/codellama-7b-instruct-awq": {
        "max_tokens": 4096, 
        "max_input_tokens": 4096, 
        "max_output_tokens": 4096, 
        "input_cost_per_token": 0.000001923, 
        "output_cost_per_token": 0.000001923, 
        "litellm_provider": "cloudflare", 
        "mode": "chat"
      },
      "voyage/voyage-01": {
        "max_tokens": 4096,
        "max_input_tokens": 4096,
        "input_cost_per_token": 0.0000001,
        "output_cost_per_token": 0.000000,
        "litellm_provider": "voyage",
        "mode": "embedding"
    },
    "voyage/voyage-lite-01": {
        "max_tokens": 4096,
        "max_input_tokens": 4096,
        "input_cost_per_token": 0.0000001,
        "output_cost_per_token": 0.000000,
        "litellm_provider": "voyage",
        "mode": "embedding"
    },
    "voyage/voyage-large-2": {
        "max_tokens": 16000,
        "max_input_tokens": 16000,
        "input_cost_per_token": 0.00000012,
        "output_cost_per_token": 0.000000,
        "litellm_provider": "voyage",
        "mode": "embedding"
    },
    "voyage/voyage-law-2": {
        "max_tokens": 16000,
        "max_input_tokens": 16000,
        "input_cost_per_token": 0.00000012,
        "output_cost_per_token": 0.000000,
        "litellm_provider": "voyage",
        "mode": "embedding"
    },
    "voyage/voyage-code-2": {
        "max_tokens": 16000,
        "max_input_tokens": 16000,
        "input_cost_per_token": 0.00000012,
        "output_cost_per_token": 0.000000,
        "litellm_provider": "voyage",
        "mode": "embedding"
    },
    "voyage/voyage-2": {
        "max_tokens": 4000,
        "max_input_tokens": 4000,
        "input_cost_per_token": 0.0000001,
        "output_cost_per_token": 0.000000,
        "litellm_provider": "voyage",
        "mode": "embedding"
    },
    "voyage/voyage-lite-02-instruct": {
        "max_tokens": 4000,
        "max_input_tokens": 4000,
        "input_cost_per_token": 0.0000001,
        "output_cost_per_token": 0.000000,
        "litellm_provider": "voyage",
        "mode": "embedding"
    },
    "databricks/databricks-meta-llama-3-1-405b-instruct": {
        "max_tokens": 128000,
        "max_input_tokens": 128000,
        "max_output_tokens": 128000, 
        "input_cost_per_token": 0.000005,
        "output_cost_per_token": 0.000015,
        "litellm_provider": "databricks",
        "mode": "chat",
        "source": "https://www.databricks.com/product/pricing/foundation-model-serving"
    },
    "databricks/databricks-meta-llama-3-1-70b-instruct": {
        "max_tokens": 128000,
        "max_input_tokens": 128000,
        "max_output_tokens": 128000, 
        "input_cost_per_token": 0.000001,
        "output_cost_per_token": 0.000003,
        "litellm_provider": "databricks",
        "mode": "chat",
        "source": "https://www.databricks.com/product/pricing/foundation-model-serving"
    },
    "databricks/databricks-dbrx-instruct": {
        "max_tokens": 32768,
        "max_input_tokens": 32768,
        "max_output_tokens": 32768, 
        "input_cost_per_token": 0.00000075,
        "output_cost_per_token": 0.00000225,
        "litellm_provider": "databricks",
        "mode": "chat",
        "source": "https://www.databricks.com/product/pricing/foundation-model-serving"
    },
    "databricks/databricks-meta-llama-3-70b-instruct": {
        "max_tokens": 8192,
        "max_input_tokens": 8192,
        "max_output_tokens": 8192, 
        "input_cost_per_token": 0.000001,
        "output_cost_per_token": 0.000003,
        "litellm_provider": "databricks",
        "mode": "chat",
        "source": "https://www.databricks.com/product/pricing/foundation-model-serving"
    },
    "databricks/databricks-llama-2-70b-chat": {
        "max_tokens": 4096,
        "max_input_tokens": 4096,
        "max_output_tokens": 4096, 
        "input_cost_per_token": 0.0000005,
        "output_cost_per_token": 0.0000015,
        "litellm_provider": "databricks",
        "mode": "chat",
        "source": "https://www.databricks.com/product/pricing/foundation-model-serving"

    },
    "databricks/databricks-mixtral-8x7b-instruct": {
        "max_tokens": 4096,
        "max_input_tokens": 4096,
        "max_output_tokens": 4096, 
        "input_cost_per_token": 0.0000005,
        "output_cost_per_token": 0.000001,
        "litellm_provider": "databricks",
        "mode": "chat",
        "source": "https://www.databricks.com/product/pricing/foundation-model-serving"
    },
    "databricks/databricks-mpt-30b-instruct": {
        "max_tokens": 8192,
        "max_input_tokens": 8192,
        "max_output_tokens": 8192, 
        "input_cost_per_token": 0.000001,
        "output_cost_per_token": 0.000001,
        "litellm_provider": "databricks",
        "mode": "chat",
        "source": "https://www.databricks.com/product/pricing/foundation-model-serving"
    },
    "databricks/databricks-mpt-7b-instruct": {
        "max_tokens": 8192,
        "max_input_tokens": 8192,
        "max_output_tokens": 8192, 
        "input_cost_per_token": 0.0000005,
        "output_cost_per_token": 0.0000005,
        "litellm_provider": "databricks",
        "mode": "chat",
        "source": "https://www.databricks.com/product/pricing/foundation-model-serving"
    },
    "databricks/databricks-bge-large-en": {
        "max_tokens": 512,
        "max_input_tokens": 512,
        "output_vector_size": 1024, 
        "input_cost_per_token": 0.0000001,
        "output_cost_per_token": 0.0,
        "litellm_provider": "databricks",
        "mode": "embedding",
        "source": "https://www.databricks.com/product/pricing/foundation-model-serving"
    }
}
"""

    content = json.loads(json_text)
    return content

    #if (
    #    os.getenv("LITELLM_LOCAL_MODEL_COST_MAP", False) == True
    #    or os.getenv("LITELLM_LOCAL_MODEL_COST_MAP", False) == "True"
    # ):
    #     import importlib.resources
    #     import json

    #     with importlib.resources.open_text(
    #         "litellm", "model_prices_and_context_window_backup.json"
    #     ) as f:
    #         content = json.load(f)
    #         return content

    # try:
    #     with requests.get(
    #         url, timeout=5
    #     ) as response:  # set a 5 second timeout for the get request
    #         response.raise_for_status()  # Raise an exception if the request is unsuccessful
    #         content = response.json()
    #         return content
    # except Exception as e:
    #     import importlib.resources
    #     import json

    #     with importlib.resources.open_text(
    #         "litellm", "model_prices_and_context_window_backup.json"
    #     ) as f:
    #         content = json.load(f)
    #         return content


model_cost = get_model_cost_map(url=model_cost_map_url)
custom_prompt_dict: Dict[str, dict] = {}


####### THREAD-SPECIFIC DATA ###################
class MyLocal(threading.local):
    def __init__(self):
        self.user = "Hello World"


_thread_context = MyLocal()


def identify(event_details):
    # Store user in thread local data
    if "user" in event_details:
        _thread_context.user = event_details["user"]


####### ADDITIONAL PARAMS ################### configurable params if you use proxy models like Helicone, map spend to org id, etc.
api_base = None
headers = None
api_version = None
organization = None
project = None
config_path = None
vertex_ai_safety_settings: Optional[dict] = None
####### COMPLETION MODELS ###################
open_ai_chat_completion_models: List = []
open_ai_text_completion_models: List = []
cohere_models: List = []
cohere_chat_models: List = []
mistral_chat_models: List = []
anthropic_models: List = []
empower_models: List = []
openrouter_models: List = []
vertex_language_models: List = []
vertex_vision_models: List = []
vertex_chat_models: List = []
vertex_code_chat_models: List = []
vertex_ai_image_models: List = []
vertex_text_models: List = []
vertex_code_text_models: List = []
vertex_embedding_models: List = []
vertex_anthropic_models: List = []
vertex_llama3_models: List = []
vertex_ai_ai21_models: List = []
vertex_mistral_models: List = []
ai21_models: List = []
ai21_chat_models: List = []
nlp_cloud_models: List = []
aleph_alpha_models: List = []
bedrock_models: List = []
fireworks_ai_models: List = []
fireworks_ai_embedding_models: List = []
deepinfra_models: List = []
perplexity_models: List = []
watsonx_models: List = []
gemini_models: List = []


def add_known_models():
    for key, value in model_cost.items():
        if value.get("litellm_provider") == "openai":
            open_ai_chat_completion_models.append(key)
        elif value.get("litellm_provider") == "text-completion-openai":
            open_ai_text_completion_models.append(key)
        elif value.get("litellm_provider") == "cohere":
            cohere_models.append(key)
        elif value.get("litellm_provider") == "cohere_chat":
            cohere_chat_models.append(key)
        elif value.get("litellm_provider") == "mistral":
            mistral_chat_models.append(key)
        elif value.get("litellm_provider") == "anthropic":
            anthropic_models.append(key)
        elif value.get("litellm_provider") == "empower":
            empower_models.append(key)
        elif value.get("litellm_provider") == "openrouter":
            openrouter_models.append(key)
        elif value.get("litellm_provider") == "vertex_ai-text-models":
            vertex_text_models.append(key)
        elif value.get("litellm_provider") == "vertex_ai-code-text-models":
            vertex_code_text_models.append(key)
        elif value.get("litellm_provider") == "vertex_ai-language-models":
            vertex_language_models.append(key)
        elif value.get("litellm_provider") == "vertex_ai-vision-models":
            vertex_vision_models.append(key)
        elif value.get("litellm_provider") == "vertex_ai-chat-models":
            vertex_chat_models.append(key)
        elif value.get("litellm_provider") == "vertex_ai-code-chat-models":
            vertex_code_chat_models.append(key)
        elif value.get("litellm_provider") == "vertex_ai-embedding-models":
            vertex_embedding_models.append(key)
        elif value.get("litellm_provider") == "vertex_ai-anthropic_models":
            key = key.replace("vertex_ai/", "")
            vertex_anthropic_models.append(key)
        elif value.get("litellm_provider") == "vertex_ai-llama_models":
            key = key.replace("vertex_ai/", "")
            vertex_llama3_models.append(key)
        elif value.get("litellm_provider") == "vertex_ai-mistral_models":
            key = key.replace("vertex_ai/", "")
            vertex_mistral_models.append(key)
        elif value.get("litellm_provider") == "vertex_ai-ai21_models":
            key = key.replace("vertex_ai/", "")
            vertex_ai_ai21_models.append(key)
        elif value.get("litellm_provider") == "vertex_ai-image-models":
            key = key.replace("vertex_ai/", "")
            vertex_ai_image_models.append(key)
        elif value.get("litellm_provider") == "ai21":
            if value.get("mode") == "chat":
                ai21_chat_models.append(key)
            else:
                ai21_models.append(key)
        elif value.get("litellm_provider") == "nlp_cloud":
            nlp_cloud_models.append(key)
        elif value.get("litellm_provider") == "aleph_alpha":
            aleph_alpha_models.append(key)
        elif value.get("litellm_provider") == "bedrock":
            bedrock_models.append(key)
        elif value.get("litellm_provider") == "deepinfra":
            deepinfra_models.append(key)
        elif value.get("litellm_provider") == "perplexity":
            perplexity_models.append(key)
        elif value.get("litellm_provider") == "watsonx":
            watsonx_models.append(key)
        elif value.get("litellm_provider") == "gemini":
            gemini_models.append(key)
        elif value.get("litellm_provider") == "fireworks_ai":
            # ignore the 'up-to', '-to-' model names -> not real models. just for cost tracking based on model params.
            if "-to-" not in key:
                fireworks_ai_models.append(key)
        elif value.get("litellm_provider") == "fireworks_ai-embedding-models":
            # ignore the 'up-to', '-to-' model names -> not real models. just for cost tracking based on model params.
            if "-to-" not in key:
                fireworks_ai_embedding_models.append(key)


add_known_models()
# known openai compatible endpoints - we'll eventually move this list to the model_prices_and_context_window.json dictionary
openai_compatible_endpoints: List = [
    "api.perplexity.ai",
    "api.endpoints.anyscale.com/v1",
    "api.deepinfra.com/v1/openai",
    "api.mistral.ai/v1",
    "codestral.mistral.ai/v1/chat/completions",
    "codestral.mistral.ai/v1/fim/completions",
    "api.groq.com/openai/v1",
    "https://integrate.api.nvidia.com/v1",
    "api.deepseek.com/v1",
    "api.together.xyz/v1",
    "app.empower.dev/api/v1",
    "inference.friendli.ai/v1",
    "api.sambanova.ai/v1",
]

# this is maintained for Exception Mapping
openai_compatible_providers: List = [
    "anyscale",
    "mistral",
    "groq",
    "nvidia_nim",
    "cerebras",
    "sambanova",
    "ai21_chat",
    "volcengine",
    "codestral",
    "deepseek",
    "deepinfra",
    "perplexity",
    "xinference",
    "together_ai",
    "fireworks_ai",
    "empower",
    "friendliai",
    "azure_ai",
    "github",
    "litellm_proxy",
]
openai_text_completion_compatible_providers: List = (
    [  # providers that support `/v1/completions`
        "together_ai",
        "fireworks_ai",
    ]
)

# well supported replicate llms
replicate_models: List = [
    # llama replicate supported LLMs
    "replicate/llama-2-70b-chat:2796ee9483c3fd7aa2e171d38f4ca12251a30609463dcfd4cd76703f22e96cdf",
    "a16z-infra/llama-2-13b-chat:2a7f981751ec7fdf87b5b91ad4db53683a98082e9ff7bfd12c8cd5ea85980a52",
    "meta/codellama-13b:1c914d844307b0588599b8393480a3ba917b660c7e9dfae681542b5325f228db",
    # Vicuna
    "replicate/vicuna-13b:6282abe6a492de4145d7bb601023762212f9ddbbe78278bd6771c8b3b2f2a13b",
    "joehoover/instructblip-vicuna13b:c4c54e3c8c97cd50c2d2fec9be3b6065563ccf7d43787fb99f84151b867178fe",
    # Flan T-5
    "daanelson/flan-t5-large:ce962b3f6792a57074a601d3979db5839697add2e4e02696b3ced4c022d4767f",
    # Others
    "replicate/dolly-v2-12b:ef0e1aefc61f8e096ebe4db6b2bacc297daf2ef6899f0f7e001ec445893500e5",
    "replit/replit-code-v1-3b:b84f4c074b807211cd75e3e8b1589b6399052125b4c27106e43d47189e8415ad",
]

clarifai_models: List = [
    "clarifai/meta.Llama-3.Llama-3-8B-Instruct",
    "clarifai/gcp.generate.gemma-1_1-7b-it",
    "clarifai/mistralai.completion.mixtral-8x22B",
    "clarifai/cohere.generate.command-r-plus",
    "clarifai/databricks.drbx.dbrx-instruct",
    "clarifai/mistralai.completion.mistral-large",
    "clarifai/mistralai.completion.mistral-medium",
    "clarifai/mistralai.completion.mistral-small",
    "clarifai/mistralai.completion.mixtral-8x7B-Instruct-v0_1",
    "clarifai/gcp.generate.gemma-2b-it",
    "clarifai/gcp.generate.gemma-7b-it",
    "clarifai/deci.decilm.deciLM-7B-instruct",
    "clarifai/mistralai.completion.mistral-7B-Instruct",
    "clarifai/gcp.generate.gemini-pro",
    "clarifai/anthropic.completion.claude-v1",
    "clarifai/anthropic.completion.claude-instant-1_2",
    "clarifai/anthropic.completion.claude-instant",
    "clarifai/anthropic.completion.claude-v2",
    "clarifai/anthropic.completion.claude-2_1",
    "clarifai/meta.Llama-2.codeLlama-70b-Python",
    "clarifai/meta.Llama-2.codeLlama-70b-Instruct",
    "clarifai/openai.completion.gpt-3_5-turbo-instruct",
    "clarifai/meta.Llama-2.llama2-7b-chat",
    "clarifai/meta.Llama-2.llama2-13b-chat",
    "clarifai/meta.Llama-2.llama2-70b-chat",
    "clarifai/openai.chat-completion.gpt-4-turbo",
    "clarifai/microsoft.text-generation.phi-2",
    "clarifai/meta.Llama-2.llama2-7b-chat-vllm",
    "clarifai/upstage.solar.solar-10_7b-instruct",
    "clarifai/openchat.openchat.openchat-3_5-1210",
    "clarifai/togethercomputer.stripedHyena.stripedHyena-Nous-7B",
    "clarifai/gcp.generate.text-bison",
    "clarifai/meta.Llama-2.llamaGuard-7b",
    "clarifai/fblgit.una-cybertron.una-cybertron-7b-v2",
    "clarifai/openai.chat-completion.GPT-4",
    "clarifai/openai.chat-completion.GPT-3_5-turbo",
    "clarifai/ai21.complete.Jurassic2-Grande",
    "clarifai/ai21.complete.Jurassic2-Grande-Instruct",
    "clarifai/ai21.complete.Jurassic2-Jumbo-Instruct",
    "clarifai/ai21.complete.Jurassic2-Jumbo",
    "clarifai/ai21.complete.Jurassic2-Large",
    "clarifai/cohere.generate.cohere-generate-command",
    "clarifai/wizardlm.generate.wizardCoder-Python-34B",
    "clarifai/wizardlm.generate.wizardLM-70B",
    "clarifai/tiiuae.falcon.falcon-40b-instruct",
    "clarifai/togethercomputer.RedPajama.RedPajama-INCITE-7B-Chat",
    "clarifai/gcp.generate.code-gecko",
    "clarifai/gcp.generate.code-bison",
    "clarifai/mistralai.completion.mistral-7B-OpenOrca",
    "clarifai/mistralai.completion.openHermes-2-mistral-7B",
    "clarifai/wizardlm.generate.wizardLM-13B",
    "clarifai/huggingface-research.zephyr.zephyr-7B-alpha",
    "clarifai/wizardlm.generate.wizardCoder-15B",
    "clarifai/microsoft.text-generation.phi-1_5",
    "clarifai/databricks.Dolly-v2.dolly-v2-12b",
    "clarifai/bigcode.code.StarCoder",
    "clarifai/salesforce.xgen.xgen-7b-8k-instruct",
    "clarifai/mosaicml.mpt.mpt-7b-instruct",
    "clarifai/anthropic.completion.claude-3-opus",
    "clarifai/anthropic.completion.claude-3-sonnet",
    "clarifai/gcp.generate.gemini-1_5-pro",
    "clarifai/gcp.generate.imagen-2",
    "clarifai/salesforce.blip.general-english-image-caption-blip-2",
]


huggingface_models: List = [
    "meta-llama/Llama-2-7b-hf",
    "meta-llama/Llama-2-7b-chat-hf",
    "meta-llama/Llama-2-13b-hf",
    "meta-llama/Llama-2-13b-chat-hf",
    "meta-llama/Llama-2-70b-hf",
    "meta-llama/Llama-2-70b-chat-hf",
    "meta-llama/Llama-2-7b",
    "meta-llama/Llama-2-7b-chat",
    "meta-llama/Llama-2-13b",
    "meta-llama/Llama-2-13b-chat",
    "meta-llama/Llama-2-70b",
    "meta-llama/Llama-2-70b-chat",
]  # these have been tested on extensively. But by default all text2text-generation and text-generation models are supported by liteLLM. - https://docs.litellm.ai/docs/providers
empower_models = [
    "empower/empower-functions",
    "empower/empower-functions-small",
]

together_ai_models: List = [
    # llama llms - chat
    "togethercomputer/llama-2-70b-chat",
    # llama llms - language / instruct
    "togethercomputer/llama-2-70b",
    "togethercomputer/LLaMA-2-7B-32K",
    "togethercomputer/Llama-2-7B-32K-Instruct",
    "togethercomputer/llama-2-7b",
    # falcon llms
    "togethercomputer/falcon-40b-instruct",
    "togethercomputer/falcon-7b-instruct",
    # alpaca
    "togethercomputer/alpaca-7b",
    # chat llms
    "HuggingFaceH4/starchat-alpha",
    # code llms
    "togethercomputer/CodeLlama-34b",
    "togethercomputer/CodeLlama-34b-Instruct",
    "togethercomputer/CodeLlama-34b-Python",
    "defog/sqlcoder",
    "NumbersStation/nsql-llama-2-7B",
    "WizardLM/WizardCoder-15B-V1.0",
    "WizardLM/WizardCoder-Python-34B-V1.0",
    # language llms
    "NousResearch/Nous-Hermes-Llama2-13b",
    "Austism/chronos-hermes-13b",
    "upstage/SOLAR-0-70b-16bit",
    "WizardLM/WizardLM-70B-V1.0",
]  # supports all together ai models, just pass in the model id e.g. completion(model="together_computer/replit_code_3b",...)


baseten_models: List = [
    "qvv0xeq",
    "q841o8w",
    "31dxrj3",
]  # FALCON 7B  # WizardLM  # Mosaic ML


# used for Cost Tracking & Token counting
# https://azure.microsoft.com/en-in/pricing/details/cognitive-services/openai-service/
# Azure returns gpt-35-turbo in their responses, we need to map this to azure/gpt-3.5-turbo for token counting
azure_llms = {
    "gpt-35-turbo": "azure/gpt-35-turbo",
    "gpt-35-turbo-16k": "azure/gpt-35-turbo-16k",
    "gpt-35-turbo-instruct": "azure/gpt-35-turbo-instruct",
}

azure_embedding_models = {
    "ada": "azure/ada",
}

petals_models = [
    "petals-team/StableBeluga2",
]

ollama_models = ["llama2"]

maritalk_models = ["maritalk"]

model_list = (
    open_ai_chat_completion_models
    + open_ai_text_completion_models
    + cohere_models
    + cohere_chat_models
    + anthropic_models
    + replicate_models
    + openrouter_models
    + huggingface_models
    + vertex_chat_models
    + vertex_text_models
    + ai21_models
    + ai21_chat_models
    + together_ai_models
    + baseten_models
    + aleph_alpha_models
    + nlp_cloud_models
    + ollama_models
    + bedrock_models
    + deepinfra_models
    + perplexity_models
    + maritalk_models
    + vertex_language_models
    + watsonx_models
    + gemini_models
)


class LlmProviders(str, Enum):
    OPENAI = "openai"
    CUSTOM_OPENAI = "custom_openai"
    TEXT_COMPLETION_OPENAI = "text-completion-openai"
    COHERE = "cohere"
    COHERE_CHAT = "cohere_chat"
    CLARIFAI = "clarifai"
    ANTHROPIC = "anthropic"
    REPLICATE = "replicate"
    HUGGINGFACE = "huggingface"
    TOGETHER_AI = "together_ai"
    OPENROUTER = "openrouter"
    VERTEX_AI = "vertex_ai"
    VERTEX_AI_BETA = "vertex_ai_beta"
    PALM = "palm"
    GEMINI = "gemini"
    AI21 = "ai21"
    BASETEN = "baseten"
    AZURE = "azure"
    AZURE_TEXT = "azure_text"
    AZURE_AI = "azure_ai"
    SAGEMAKER = "sagemaker"
    SAGEMAKER_CHAT = "sagemaker_chat"
    BEDROCK = "bedrock"
    VLLM = "vllm"
    NLP_CLOUD = "nlp_cloud"
    PETALS = "petals"
    OOBABOOGA = "oobabooga"
    OLLAMA = "ollama"
    OLLAMA_CHAT = "ollama_chat"
    DEEPINFRA = "deepinfra"
    PERPLEXITY = "perplexity"
    ANYSCALE = "anyscale"
    MISTRAL = "mistral"
    GROQ = "groq"
    NVIDIA_NIM = "nvidia_nim"
    CEREBRAS = "cerebras"
    AI21_CHAT = "ai21_chat"
    VOLCENGINE = "volcengine"
    CODESTRAL = "codestral"
    TEXT_COMPLETION_CODESTRAL = "text-completion-codestral"
    DEEPSEEK = "deepseek"
    SAMBANOVA = "sambanova"
    MARITALK = "maritalk"
    VOYAGE = "voyage"
    CLOUDFLARE = "cloudflare"
    XINFERENCE = "xinference"
    FIREWORKS_AI = "fireworks_ai"
    FRIENDLIAI = "friendliai"
    WATSONX = "watsonx"
    TRITON = "triton"
    PREDIBASE = "predibase"
    DATABRICKS = "databricks"
    EMPOWER = "empower"
    GITHUB = "github"
    CUSTOM = "custom"
    LITELLM_PROXY = "litellm_proxy"


provider_list: List[Union[LlmProviders, str]] = list(LlmProviders)


models_by_provider: dict = {
    "openai": open_ai_chat_completion_models + open_ai_text_completion_models,
    "cohere": cohere_models + cohere_chat_models,
    "cohere_chat": cohere_chat_models,
    "anthropic": anthropic_models,
    "replicate": replicate_models,
    "huggingface": huggingface_models,
    "together_ai": together_ai_models,
    "baseten": baseten_models,
    "openrouter": openrouter_models,
    "vertex_ai": vertex_chat_models
    + vertex_text_models
    + vertex_anthropic_models
    + vertex_vision_models
    + vertex_language_models,
    "ai21": ai21_models,
    "bedrock": bedrock_models,
    "petals": petals_models,
    "ollama": ollama_models,
    "deepinfra": deepinfra_models,
    "perplexity": perplexity_models,
    "maritalk": maritalk_models,
    "watsonx": watsonx_models,
    "gemini": gemini_models,
    "fireworks_ai": fireworks_ai_models + fireworks_ai_embedding_models,
}

# mapping for those models which have larger equivalents
longer_context_model_fallback_dict: dict = {
    # openai chat completion models
    "gpt-3.5-turbo": "gpt-3.5-turbo-16k",
    "gpt-3.5-turbo-0301": "gpt-3.5-turbo-16k-0301",
    "gpt-3.5-turbo-0613": "gpt-3.5-turbo-16k-0613",
    "gpt-4": "gpt-4-32k",
    "gpt-4-0314": "gpt-4-32k-0314",
    "gpt-4-0613": "gpt-4-32k-0613",
    # anthropic
    "claude-instant-1": "claude-2",
    "claude-instant-1.2": "claude-2",
    # vertexai
    "chat-bison": "chat-bison-32k",
    "chat-bison@001": "chat-bison-32k",
    "codechat-bison": "codechat-bison-32k",
    "codechat-bison@001": "codechat-bison-32k",
    # openrouter
    "openrouter/openai/gpt-3.5-turbo": "openrouter/openai/gpt-3.5-turbo-16k",
    "openrouter/anthropic/claude-instant-v1": "openrouter/anthropic/claude-2",
}

####### EMBEDDING MODELS ###################
open_ai_embedding_models: List = ["text-embedding-ada-002"]
cohere_embedding_models: List = [
    "embed-english-v3.0",
    "embed-english-light-v3.0",
    "embed-multilingual-v3.0",
    "embed-english-v2.0",
    "embed-english-light-v2.0",
    "embed-multilingual-v2.0",
]
bedrock_embedding_models: List = [
    "amazon.titan-embed-text-v1",
    "cohere.embed-english-v3",
    "cohere.embed-multilingual-v3",
]

all_embedding_models = (
    open_ai_embedding_models
    + cohere_embedding_models
    + bedrock_embedding_models
    + vertex_embedding_models
    + fireworks_ai_embedding_models
)

####### IMAGE GENERATION MODELS ###################
openai_image_generation_models = ["dall-e-2", "dall-e-3"]

from .timeout import timeout
from .cost_calculator import completion_cost
from litellm.litellm_core_utils.litellm_logging import Logging
from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider
from litellm.litellm_core_utils.core_helpers import remove_index_from_tool_calls
from litellm.litellm_core_utils.token_counter import get_modified_max_tokens
from .utils import (
    client,
    exception_type,
    get_optional_params,
    get_response_string,
    modify_integration,
    token_counter,
    create_pretrained_tokenizer,
    create_tokenizer,
    supports_function_calling,
    supports_response_schema,
    supports_parallel_function_calling,
    supports_vision,
    supports_system_messages,
    get_litellm_params,
    acreate,
    get_model_list,
    get_max_tokens,
    get_model_info,
    register_prompt_template,
    validate_environment,
    check_valid_key,
    register_model,
    encode,
    decode,
    _calculate_retry_after,
    _should_retry,
    get_supported_openai_params,
    get_api_base,
    get_first_chars_messages,
    ModelResponse,
    EmbeddingResponse,
    ImageResponse,
    TranscriptionResponse,
    TextCompletionResponse,
    get_provider_fields,
    ModelResponseListIterator,
)

ALL_LITELLM_RESPONSE_TYPES = [
    ModelResponse,
    EmbeddingResponse,
    ImageResponse,
    TranscriptionResponse,
    TextCompletionResponse,
]

from .types.utils import ImageObject
from .llms.custom_llm import CustomLLM
from .llms.huggingface_restapi import HuggingfaceConfig
from .llms.anthropic.chat import AnthropicConfig
from .llms.anthropic.completion import AnthropicTextConfig
from .llms.databricks.chat import DatabricksConfig, DatabricksEmbeddingConfig
from .llms.predibase import PredibaseConfig
from .llms.replicate import ReplicateConfig
from .llms.cohere.completion import CohereConfig
from .llms.clarifai import ClarifaiConfig
from .llms.AI21.completion import AI21Config
from .llms.AI21.chat import AI21ChatConfig
from .llms.together_ai.chat import TogetherAIConfig
from .llms.cloudflare import CloudflareConfig
from .llms.palm import PalmConfig
from .llms.gemini import GeminiConfig
from .llms.nlp_cloud import NLPCloudConfig
from .llms.aleph_alpha import AlephAlphaConfig
from .llms.petals import PetalsConfig
from .llms.vertex_ai_and_google_ai_studio.gemini.vertex_and_google_ai_studio_gemini import (
    VertexGeminiConfig,
    GoogleAIStudioGeminiConfig,
    VertexAIConfig,
)
from .llms.vertex_ai_and_google_ai_studio.vertex_embeddings.embedding_handler import (
    VertexAITextEmbeddingConfig,
)
from .llms.vertex_ai_and_google_ai_studio.vertex_ai_anthropic import (
    VertexAIAnthropicConfig,
)
from .llms.vertex_ai_and_google_ai_studio.vertex_ai_partner_models.llama3.transformation import (
    VertexAILlama3Config,
)
from .llms.vertex_ai_and_google_ai_studio.vertex_ai_partner_models.ai21.transformation import (
    VertexAIAi21Config,
)

from .llms.sagemaker.sagemaker import SagemakerConfig
from .llms.ollama import OllamaConfig
from .llms.ollama_chat import OllamaChatConfig
from .llms.maritalk import MaritTalkConfig
from .llms.bedrock.chat.invoke_handler import (
    AmazonCohereChatConfig,
    AmazonConverseConfig,
    bedrock_tool_name_mappings,
)
from .llms.bedrock.chat.converse_handler import (
    BEDROCK_CONVERSE_MODELS,
)
from .llms.bedrock.common_utils import (
    AmazonTitanConfig,
    AmazonAI21Config,
    AmazonAnthropicConfig,
    AmazonAnthropicClaude3Config,
    AmazonCohereConfig,
    AmazonLlamaConfig,
    AmazonStabilityConfig,
    AmazonMistralConfig,
    AmazonBedrockGlobalConfig,
)
from .llms.bedrock.embed.amazon_titan_g1_transformation import AmazonTitanG1Config
from .llms.bedrock.embed.amazon_titan_multimodal_transformation import (
    AmazonTitanMultimodalEmbeddingG1Config,
)
from .llms.bedrock.embed.amazon_titan_v2_transformation import (
    AmazonTitanV2Config,
)
from .llms.bedrock.embed.cohere_transformation import BedrockCohereEmbeddingConfig
from .llms.OpenAI.openai import (
    OpenAIConfig,
    OpenAITextCompletionConfig,
    MistralEmbeddingConfig,
    DeepInfraConfig,
    GroqConfig,
)
from .llms.azure_ai.chat.transformation import AzureAIStudioConfig
from .llms.mistral.mistral_chat_transformation import MistralConfig
from .llms.OpenAI.chat.o1_transformation import (
    OpenAIO1Config,
)
from .llms.OpenAI.chat.gpt_transformation import (
    OpenAIGPTConfig,
)
from .llms.nvidia_nim import NvidiaNimConfig
from .llms.cerebras.chat import CerebrasConfig
from .llms.sambanova.chat import SambanovaConfig
from .llms.AI21.chat import AI21ChatConfig
from .llms.fireworks_ai.chat.fireworks_ai_transformation import FireworksAIConfig
from .llms.fireworks_ai.embed.fireworks_ai_transformation import (
    FireworksAIEmbeddingConfig,
)
from .llms.volcengine import VolcEngineConfig
from .llms.text_completion_codestral import MistralTextCompletionConfig
from .llms.AzureOpenAI.azure import (
    AzureOpenAIConfig,
    AzureOpenAIError,
    AzureOpenAIAssistantsAPIConfig,
)
from .llms.watsonx import IBMWatsonXAIConfig
from .main import *  # type: ignore
from .integrations import *
from .exceptions import (
    AuthenticationError,
    InvalidRequestError,
    BadRequestError,
    NotFoundError,
    RateLimitError,
    ServiceUnavailableError,
    OpenAIError,
    ContextWindowExceededError,
    ContentPolicyViolationError,
    BudgetExceededError,
    APIError,
    Timeout,
    APIConnectionError,
    UnsupportedParamsError,
    APIResponseValidationError,
    UnprocessableEntityError,
    InternalServerError,
    JSONSchemaValidationError,
    LITELLM_EXCEPTION_TYPES,
    MockException,
)
from .budget_manager import BudgetManager
from .proxy.proxy_cli import run_server
from .router import Router
from .assistants.main import *
from .batches.main import *
from .rerank_api.main import *
from .fine_tuning.main import *
from .files.main import *
from .scheduler import *
from .cost_calculator import response_cost_calculator, cost_per_token

### ADAPTERS ###
from .types.adapter import AdapterItem

adapters: List[AdapterItem] = []

### CUSTOM LLMs ###
from .types.llms.custom_llm import CustomLLMItem
from .types.utils import GenericStreamingChunk

custom_provider_map: List[CustomLLMItem] = []
_custom_providers: List[str] = (
    []
)  # internal helper util, used to track names of custom providers
