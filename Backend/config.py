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

    def set_logger(self, logger: CustomLogger):
        self.logger = logger

    def read_config_value(self, key_name:str):
        return self._get_config_value(key_name)

    def _get_config_value(self, key_name:str)-> str:
        value = os.getenv(key_name, None)
        if value is None:
            if self.logger:
                self.logger.error(f"Necessary value {key_name} couldn't be found in environment")
            raise Exception(f"Necessary value {key_name} couldn't be found in environment")
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
            with open(self.config_path, 'r') as file:
                self.config = yaml.safe_load(file)
            self.logger.info(f"Bot configuration loaded successfully from {self.config_path}")
        except FileNotFoundError:
            self.logger.error(f"Configuration file not found: {self.config_path}")
            raise
        except yaml.YAMLError as e:
            self.logger.error(f"Error parsing YAML configuration: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Unexpected error loading configuration: {e}")
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
            config_reader = Config_Reader(None)

            cls.logger = CustomLogger()
            config_reader.set_logger(cls.logger)

            try:
                cls.ENV = config_reader.read_config_value("ENVIRONMENT")
                cls.GROQ_API_KEY = config_reader.read_config_value("GROQ_API_KEY")
                
                # Load bot configuration
                config_path = os.path.join(os.path.dirname(__file__), 'bot_config.yaml')
                cls.bot_config = BotConfig(config_path, cls.logger)
                
                cls.logger.info("Config values loaded successfully")
                cls.logger.info(f"Connected to {cls.ENV}")
                cls.logger.info(f"RL Mode: {'Enabled' if cls.bot_config.is_rl_enabled() else 'Disabled'}")
                cls._initialised = True
            except Exception as e:
                cls.logger.error(f"Error loading config values: {e}")
                raise e












