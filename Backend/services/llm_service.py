"""
LangChain-based LLM Service - Modular provider support
Supports: OpenAI, Azure OpenAI, Groq, Anthropic, Cohere
"""
import os
from typing import Optional, List, Dict
from langchain_openai import ChatOpenAI, AzureChatOpenAI
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.language_models.chat_models import BaseChatModel

try:
    from langchain_anthropic import ChatAnthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

try:
    from langchain_cohere import ChatCohere
    COHERE_AVAILABLE = True
except ImportError:
    COHERE_AVAILABLE = False


class LLMService:
    """Unified LLM service supporting multiple providers via LangChain"""
    
    SUPPORTED_PROVIDERS = {
        'openai': 'OpenAI',
        'azure': 'Azure OpenAI',
        'groq': 'Groq',
        'anthropic': 'Anthropic',
        'cohere': 'Cohere'
    }
    
    def __init__(
        self,
        provider: str,
        model_name: str,
        temperature: float = 0.7,
        max_tokens: int = 2000,
        logger=None
    ):
        """
        Initialize LLM service with specified provider
        
        Args:
            provider: Provider name (openai, azure, groq, anthropic, cohere)
            model_name: Model identifier
            temperature: Sampling temperature (0.0-2.0)
            max_tokens: Maximum tokens in response
            logger: Logger instance
        """
        self.provider = provider.lower()
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.logger = logger
        
        if self.provider not in self.SUPPORTED_PROVIDERS:
            raise ValueError(
                f"Unsupported provider: {provider}. "
                f"Supported: {list(self.SUPPORTED_PROVIDERS.keys())}"
            )
        
        self.client = self._initialize_client()
        
        if self.logger:
            self.logger.info(f"LLMService initialized - Provider: {self.SUPPORTED_PROVIDERS[self.provider]}")
            self.logger.info(f"Model: {self.model_name}, Temperature: {self.temperature}")
    
    def _initialize_client(self) -> BaseChatModel:
        """Initialize the appropriate LangChain client based on provider"""
        
        if self.provider == 'openai':
            api_key = os.getenv('OPENAI_API_KEY')
            if not api_key:
                raise ValueError("OPENAI_API_KEY not found in environment")
            
            return ChatOpenAI(
                api_key=api_key,
                model_name=self.model_name,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
        
        elif self.provider == 'azure':
            api_key = os.getenv('AZURE_OPENAI_API_KEY')
            endpoint = os.getenv('AZURE_OPENAI_ENDPOINT')
            api_version = os.getenv('AZURE_OPENAI_API_VERSION', '2024-02-15-preview')
            deployment_name = os.getenv('AZURE_OPENAI_DEPLOYMENT_NAME', self.model_name)
            
            if not api_key or not endpoint:
                raise ValueError(
                    "AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT required"
                )
            
            return AzureChatOpenAI(
                azure_endpoint=endpoint,
                api_key=api_key,
                api_version=api_version,
                deployment_name=deployment_name,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
        
        elif self.provider == 'groq':
            api_key = os.getenv('GROQ_API_KEY')
            if not api_key:
                raise ValueError("GROQ_API_KEY not found in environment")
            
            return ChatGroq(
                api_key=api_key,
                model_name=self.model_name,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
        
        elif self.provider == 'anthropic':
            if not ANTHROPIC_AVAILABLE:
                raise ValueError(
                    "Anthropic support not installed. "
                    "Run: pip install langchain-anthropic"
                )
            api_key = os.getenv('ANTHROPIC_API_KEY')
            if not api_key:
                raise ValueError("ANTHROPIC_API_KEY not found in environment")
            
            return ChatAnthropic(
                api_key=api_key,
                model_name=self.model_name,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
        
        elif self.provider == 'cohere':
            if not COHERE_AVAILABLE:
                raise ValueError(
                    "Cohere support not installed. "
                    "Run: pip install langchain-cohere"
                )
            api_key = os.getenv('COHERE_API_KEY')
            if not api_key:
                raise ValueError("COHERE_API_KEY not found in environment")
            
            return ChatCohere(
                cohere_api_key=api_key,
                model=self.model_name,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
    
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> str:
        """
        Generate a response using the configured LLM
        
        Args:
            prompt: User prompt/message
            system_prompt: Optional system message
            temperature: Override default temperature
            max_tokens: Override default max_tokens
            
        Returns:
            Generated text response
            
        Raises:
            ValueError: If prompt is None or empty
            RuntimeError: If LLM service fails to generate response
        """
        try:
            # Input validation
            if not prompt or not prompt.strip():
                error_msg = "Prompt cannot be None or empty"
                if self.logger:
                    self.logger.error(f"LLMService.generate validation error: {error_msg}")
                raise ValueError(error_msg)
            
            if self.logger:
                self.logger.info("=" * 50)
                self.logger.info(f"LLMService.generate called - Provider: {self.provider}")
                self.logger.info(f"Prompt length: {len(prompt)} chars")
                self.logger.info(f"System prompt: {'present (' + str(len(system_prompt)) + ' chars)' if system_prompt else 'none'}")
                self.logger.info(f"Temperature: {temperature if temperature is not None else self.temperature}")
                self.logger.info(f"Max tokens: {max_tokens if max_tokens is not None else self.max_tokens}")
            
            messages = []
            
            if system_prompt:
                messages.append(SystemMessage(content=system_prompt))
            
            messages.append(HumanMessage(content=prompt))
            
            if self.logger:
                self.logger.info(f"Total messages: {len(messages)}")
            
            # Create client with override parameters if provided
            if temperature is not None or max_tokens is not None:
                if self.logger:
                    self.logger.info("Using client with parameter overrides")
                client = self._get_client_with_overrides(temperature, max_tokens)
            else:
                if self.logger:
                    self.logger.info("Using default client configuration")
                client = self.client
            
            if self.logger:
                self.logger.info("Invoking LLM API...")
            
            response = client.invoke(messages)
            
            if not response or not response.content:
                error_msg = "LLM returned empty response"
                if self.logger:
                    self.logger.error(error_msg)
                raise RuntimeError(error_msg)
            
            if self.logger:
                self.logger.info(f"Response received - Length: {len(response.content)} chars")
                self.logger.info("=" * 50)
            
            return response.content
            
        except ValueError as ve:
            if self.logger:
                self.logger.error(f"Validation error in LLMService.generate: {ve}")
            raise
        except Exception as e:
            if self.logger:
                self.logger.error("=" * 50)
                self.logger.error(f"Critical error in LLMService.generate: {e}")
                self.logger.error(f"Error type: {type(e).__name__}")
                self.logger.error(f"Provider: {self.provider}")
                self.logger.error(f"Model: {self.model_name}")
                self.logger.error("=" * 50)
            raise
    
    def generate_with_history(
        self,
        prompt: str,
        history: List[Dict[str, str]],
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> str:
        """
        Generate response with conversation history
        
        Args:
            prompt: Current user prompt
            history: List of {"role": "user"|"assistant", "content": "..."}
            system_prompt: Optional system message
            temperature: Override default temperature
            max_tokens: Override default max_tokens
            
        Returns:
            Generated text response
        """
        try:
            messages = []
            
            if system_prompt:
                messages.append(SystemMessage(content=system_prompt))
            
            # Add history
            for msg in history:
                if msg["role"] == "user":
                    messages.append(HumanMessage(content=msg["content"]))
                elif msg["role"] == "assistant":
                    messages.append(AIMessage(content=msg["content"]))
            
            # Add current prompt
            messages.append(HumanMessage(content=prompt))
            
            # Create client with override parameters if provided
            if temperature is not None or max_tokens is not None:
                client = self._get_client_with_overrides(temperature, max_tokens)
            else:
                client = self.client
            
            if self.logger:
                self.logger.info(f"Generating with history - Messages: {len(messages)}")
            
            response = client.invoke(messages)
            
            if self.logger:
                self.logger.info(f"Response generated - Length: {len(response.content)} chars")
            
            return response.content
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error generating with history: {e}")
            raise
    
    def _get_client_with_overrides(
        self,
        temperature: Optional[float],
        max_tokens: Optional[int]
    ) -> BaseChatModel:
        """Create a new client instance with override parameters"""
        temp = temperature if temperature is not None else self.temperature
        tokens = max_tokens if max_tokens is not None else self.max_tokens
        
        # Create new instance with overrides
        if self.provider == 'openai':
            return ChatOpenAI(
                api_key=os.getenv('OPENAI_API_KEY'),
                model_name=self.model_name,
                temperature=temp,
                max_tokens=tokens
            )
        elif self.provider == 'azure':
            return AzureChatOpenAI(
                azure_endpoint=os.getenv('AZURE_OPENAI_ENDPOINT'),
                api_key=os.getenv('AZURE_OPENAI_API_KEY'),
                api_version=os.getenv('AZURE_OPENAI_API_VERSION', '2024-02-15-preview'),
                deployment_name=os.getenv('AZURE_OPENAI_DEPLOYMENT_NAME', self.model_name),
                temperature=temp,
                max_tokens=tokens
            )
        elif self.provider == 'groq':
            return ChatGroq(
                api_key=os.getenv('GROQ_API_KEY'),
                model_name=self.model_name,
                temperature=temp,
                max_tokens=tokens
            )
        elif self.provider == 'anthropic':
            if not ANTHROPIC_AVAILABLE:
                raise ValueError("Anthropic support not installed")
            return ChatAnthropic(
                api_key=os.getenv('ANTHROPIC_API_KEY'),
                model_name=self.model_name,
                temperature=temp,
                max_tokens=tokens
            )
        elif self.provider == 'cohere':
            if not COHERE_AVAILABLE:
                raise ValueError("Cohere support not installed")
            return ChatCohere(
                cohere_api_key=os.getenv('COHERE_API_KEY'),
                model=self.model_name,
                temperature=temp,
                max_tokens=tokens
            )
    
    @staticmethod
    def from_env(logger=None) -> 'LLMService':
        """
        Create LLMService instance from environment variables
        
        Required env vars:
            - LLM_PROVIDER: openai, azure, groq, anthropic, or cohere
            - LLM_MODEL_NAME: Model identifier
            - LLM_TEMPERATURE: Temperature (default 0.7)
            - LLM_MAX_TOKENS: Max tokens (default 2000)
            
        Provider-specific env vars:
            OpenAI: OPENAI_API_KEY
            Azure: AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, 
                   AZURE_OPENAI_API_VERSION (optional), 
                   AZURE_OPENAI_DEPLOYMENT_NAME (optional)
            Groq: GROQ_API_KEY
            Anthropic: ANTHROPIC_API_KEY
            Cohere: COHERE_API_KEY
            
        Raises:
            ValueError: If required environment variables are missing
            RuntimeError: If service initialization fails
        """
        try:
            if logger:
                logger.info("=" * 60)
                logger.info("Creating LLMService from environment variables")
            
            provider = os.getenv('LLM_PROVIDER')
            if not provider:
                error_msg = "LLM_PROVIDER not set in environment"
                if logger:
                    logger.error(error_msg)
                raise ValueError(error_msg)
            
            if logger:
                logger.info(f"Provider: {provider}")
            
            model_name = os.getenv('LLM_MODEL_NAME')
            if not model_name:
                error_msg = "LLM_MODEL_NAME not set in environment"
                if logger:
                    logger.error(error_msg)
                raise ValueError(error_msg)
            
            if logger:
                logger.info(f"Model: {model_name}")
            
            # Parse and validate temperature
            try:
                temperature = float(os.getenv('LLM_TEMPERATURE', '0.7'))
                if temperature < 0.0 or temperature > 2.0:
                    if logger:
                        logger.warning(f"Temperature {temperature} outside normal range [0.0, 2.0], using 0.7")
                    temperature = 0.7
            except ValueError as e:
                if logger:
                    logger.warning(f"Invalid LLM_TEMPERATURE value, using default 0.7: {e}")
                temperature = 0.7
            
            if logger:
                logger.info(f"Temperature: {temperature}")
            
            # Parse and validate max_tokens
            try:
                max_tokens = int(os.getenv('LLM_MAX_TOKENS', '2000'))
                if max_tokens < 1:
                    if logger:
                        logger.warning(f"Max tokens {max_tokens} too low, using 2000")
                    max_tokens = 2000
            except ValueError as e:
                if logger:
                    logger.warning(f"Invalid LLM_MAX_TOKENS value, using default 2000: {e}")
                max_tokens = 2000
            
            if logger:
                logger.info(f"Max Tokens: {max_tokens}")
            
            # Azure-specific validation
            if provider.lower() == 'azure':
                endpoint = os.getenv('AZURE_OPENAI_ENDPOINT')
                if logger:
                    logger.info(f"Azure Endpoint: {endpoint if endpoint else 'NOT SET'}")
                if not endpoint:
                    error_msg = "AZURE_OPENAI_ENDPOINT required for Azure provider"
                    if logger:
                        logger.error(error_msg)
                    raise ValueError(error_msg)
            
            if logger:
                logger.info("Creating LLMService instance...")
            
            service = LLMService(
                provider=provider,
                model_name=model_name,
                temperature=temperature,
                max_tokens=max_tokens,
                logger=logger
            )
            
            if logger:
                logger.info("LLMService created successfully")
                logger.info("=" * 60)
            
            return service
            
        except ValueError as ve:
            if logger:
                logger.error(f"Configuration error: {ve}")
            raise
        except Exception as e:
            if logger:
                logger.error(f"Failed to create LLMService: {e}")
                logger.error(f"Error type: {type(e).__name__}")
            raise RuntimeError(f"Failed to initialize LLMService: {e}") from e
