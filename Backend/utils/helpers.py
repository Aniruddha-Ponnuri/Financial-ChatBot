import re
from .logger import CustomLogger

# Initialize logger
logger = CustomLogger()

def format_message_as_html(message):
    """
    Format the message into HTML-friendly format
    Converts newlines to <br> tags
    """
    logger.info(f"Formatting message as HTML (input length: {len(message)} chars)")
    formatted_message = message.replace("\n", "<br>")
    logger.info(f"HTML formatting complete (output length: {len(formatted_message)} chars)")
    return formatted_message


def remove_html_tags(text):
    """
    Remove HTML tags from text for summarization
    Uses regex to strip all HTML tags
    """
    logger.info(f"Removing HTML tags from text (input length: {len(text)} chars)")
    clean = re.compile('<.*?>')  # Regex to remove HTML tags
    cleaned_text = re.sub(clean, '', text)
    logger.info(f"HTML tags removed (output length: {len(cleaned_text)} chars)")
    return cleaned_text
