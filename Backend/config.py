import os
import yaml
from dotenv import dotenv_values, find_dotenv, load_dotenv
from utils.logger import CustomLogger 


dotenv_path = find_dotenv()
env_dict = dotenv_values(dotenv_path)
env = env_dict.get("ENVIRONMENT")
if env in ("PROD", "DEV"):
    load_dotenv(dotenv_path, override=True)


class Config_Reader:
    def __init__(self, logger: CustomLogger):
        self.logger = logger
        if logger:
            logger.info("Config_Reader initialized")

    def set_logger(self, logger: CustomLogger):
        self.logger = logger
        if logger:
            logger.info("Logger set for Config_Reader")

    def read_config_value(self, key_name:str):
        if self.logger:
            self.logger.info(f"Reading config value for key: {key_name}")
        return self._get_config_value(key_name)

    def _get_config_value(self, key_name:str)-> str:
        value = os.getenv(key_name, None)
        if value is None:
            if self.logger:
                self.logger.error(f"Necessary value {key_name} couldn't be found in environment")
            raise Exception(f"Necessary value {key_name} couldn't be found in environment")
        if self.logger:
            self.logger.info(f"Config value retrieved for {key_name}: {'***' if 'KEY' in key_name or 'PASSWORD' in key_name else value}")
        return value


class BotConfig:
    """Configuration loader for bot_config.yaml"""
    def __init__(self, config_path: str, logger: CustomLogger):
        self.logger = logger
        self.config_path = config_path
        self.config = {}
        self._load_config()
    
    def _load_config(self):
        """Load configuration from YAML file"""
        try:
            self.logger.info(f"Loading bot configuration from: {self.config_path}")
            with open(self.config_path, 'r') as file:
                self.config = yaml.safe_load(file)
            
            # Log configuration sections loaded
            config_sections = list(self.config.keys()) if self.config else []
            self.logger.info(f"Configuration sections loaded: {config_sections}")
            self.logger.info(f"Bot configuration loaded successfully from {self.config_path}")
            
        except FileNotFoundError:
            self.logger.error(f"Configuration file not found: {self.config_path}")
            raise
        except yaml.YAMLError as e:
            self.logger.error(f"Error parsing YAML configuration: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Unexpected error loading configuration: {e}")
            self.logger.error(f"Error type: {type(e).__name__}")
            raise
    
    def get(self, key_path: str, default=None):
        """
        Get configuration value using dot notation
        Example: get('rl_config.n_candidates') returns the n_candidates value
        """
        keys = key_path.split('.')
        value = self.config
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                self.logger.warning(f"Configuration key '{key_path}' not found, using default: {default}")
                return default
        
        self.logger.info(f"Retrieved config value for '{key_path}': {value if not isinstance(value, (dict, list)) else type(value).__name__}")
        return value
    
    def get_prompt(self, prompt_name: str) -> str:
        """Get a prompt from the prompts section"""
        return self.get(f'prompts.{prompt_name}', '')
    
    def get_rl_config(self, key: str, default=None):
        """Get RL configuration value"""
        return self.get(f'rl_config.{key}', default)
    
    def get_model_config(self, key: str, default=None):
        """Get model configuration value"""
        return self.get(f'model.{key}', default)
    
    def is_rl_enabled(self) -> bool:
        """Check if RL is enabled"""
        return self.get('rl_config.enabled', False)


class DefaultConfig:
    _initialised = False

    @classmethod
    def initialise(cls):
        if not cls._initialised:
            cls.logger = CustomLogger()
            cls.logger.info("=" * 80)
            cls.logger.info("Initializing DefaultConfig")
            
            config_reader = Config_Reader(None)
            config_reader.set_logger(cls.logger)

            try:
                cls.logger.info("Reading environment configuration")
                cls.ENV = config_reader.read_config_value("ENVIRONMENT")
                cls.LLM_PROVIDER = config_reader.read_config_value("LLM_PROVIDER")
                
                # Read API key based on provider
                provider = cls.LLM_PROVIDER.lower()
                if provider == "groq":
                    cls.API_KEY = config_reader.read_config_value("GROQ_API_KEY")
                elif provider == "openai":
                    cls.API_KEY = config_reader.read_config_value("OPENAI_API_KEY")
                elif provider == "azure":
                    cls.API_KEY = config_reader.read_config_value("AZURE_OPENAI_API_KEY")
                elif provider == "anthropic":
                    cls.API_KEY = config_reader.read_config_value("ANTHROPIC_API_KEY")
                elif provider == "cohere":
                    cls.API_KEY = config_reader.read_config_value("COHERE_API_KEY")
                else:
                    cls.logger.warning(f"Unknown provider: {provider}")
                    cls.API_KEY = None
                
                # Load bot configuration
                config_path = os.path.join(os.path.dirname(__file__), 'bot_config.yaml')
                cls.logger.info(f"Bot config path: {config_path}")
                cls.bot_config = BotConfig(config_path, cls.logger)
                
                cls.logger.info("Config values loaded successfully")
                cls.logger.info(f"Environment: {cls.ENV}")
                cls.logger.info(f"LLM Provider: {cls.LLM_PROVIDER}")
                cls.logger.info(f"RL Mode: {'Enabled' if cls.bot_config.is_rl_enabled() else 'Disabled'}")
                cls._initialised = True
                cls.logger.info("DefaultConfig initialization complete")
                cls.logger.info("=" * 80)
                
            except Exception as e:
                cls.logger.error(f"Error loading config values: {e}")
                cls.logger.error(f"Error type: {type(e).__name__}")
                cls.logger.error("=" * 80)
                raise e












