"""
Test script to verify LLM Service functionality
Tests all provider configurations
"""
import os
import sys
from dotenv import load_dotenv

# Add parent directory to path
sys.path.insert(0, os.path.dirname(__file__))

load_dotenv()

def test_provider(provider_name):
    """Test a specific provider configuration"""
    print(f"\n{'='*60}")
    print(f"Testing {provider_name.upper()} Provider")
    print('='*60)
    
    try:
        from services.llm_service import LLMService
        from utils.logger import CustomLogger
        
        logger = CustomLogger()
        
        # Create service from environment
        llm_service = LLMService.from_env(logger=logger)
        
        print(f"✓ {provider_name} service initialized successfully")
        print(f"  Model: {llm_service.model_name}")
        print(f"  Temperature: {llm_service.temperature}")
        print(f"  Max Tokens: {llm_service.max_tokens}")
        
        # Test generation
        print("\nTesting generation...")
        response = llm_service.generate(
            prompt="What is 2+2? Answer with just the number.",
            temperature=0.1,
            max_tokens=10
        )
        
        print(f"✓ Generation successful")
        print(f"  Response: {response[:100]}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error testing {provider_name}: {e}")
        return False

def main():
    print("="*60)
    print("  LLM Service Test Suite")
    print("="*60)
    
    # Check environment configuration
    provider = os.getenv('LLM_PROVIDER')
    model = os.getenv('LLM_MODEL_NAME')
    
    if not provider:
        print("\n✗ ERROR: LLM_PROVIDER not set in .env")
        print("Please configure your .env file")
        sys.exit(1)
    
    if not model:
        print("\n✗ ERROR: LLM_MODEL_NAME not set in .env")
        sys.exit(1)
    
    print(f"\nEnvironment Configuration:")
    print(f"  Provider: {provider}")
    print(f"  Model: {model}")
    print(f"  Temperature: {os.getenv('LLM_TEMPERATURE', '0.7')}")
    print(f"  Max Tokens: {os.getenv('LLM_MAX_TOKENS', '2000')}")
    
    # Check API key
    api_key_vars = {
        'openai': 'OPENAI_API_KEY',
        'azure': 'AZURE_OPENAI_API_KEY',
        'groq': 'GROQ_API_KEY',
        'anthropic': 'ANTHROPIC_API_KEY',
        'cohere': 'COHERE_API_KEY'
    }
    
    required_key = api_key_vars.get(provider.lower())
    if required_key:
        key_value = os.getenv(required_key)
        if key_value:
            print(f"  API Key: {required_key[:20]}...{required_key[-10:]} (set)")
        else:
            print(f"\n✗ ERROR: {required_key} not set in .env")
            sys.exit(1)
    
    # Azure-specific checks
    if provider.lower() == 'azure':
        endpoint = os.getenv('AZURE_OPENAI_ENDPOINT')
        if not endpoint:
            print("\n✗ ERROR: AZURE_OPENAI_ENDPOINT not set in .env")
            sys.exit(1)
        print(f"  Endpoint: {endpoint}")
        print(f"  API Version: {os.getenv('AZURE_OPENAI_API_VERSION', '2024-02-15-preview')}")
        deployment = os.getenv('AZURE_OPENAI_DEPLOYMENT_NAME', model)
        print(f"  Deployment: {deployment}")
    
    # Run test
    success = test_provider(provider)
    
    print("\n" + "="*60)
    if success:
        print("  ✓ ALL TESTS PASSED")
        print("="*60)
        print("\nYour LLM service is configured correctly!")
        print("You can now run: python app.py")
    else:
        print("  ✗ TESTS FAILED")
        print("="*60)
        print("\nPlease check your configuration and try again.")
        sys.exit(1)

if __name__ == "__main__":
    main()
