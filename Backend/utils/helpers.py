import re

def format_message_as_html(message):
    """
    Format the message into HTML-friendly format
    Converts newlines to <br> tags
    """
    formatted_message = message.replace("\n", "<br>")
    return formatted_message


def remove_html_tags(text):
    """
    Remove HTML tags from text for summarization
    Uses regex to strip all HTML tags
    """
    clean = re.compile('<.*?>')  # Regex to remove HTML tags
    return re.sub(clean, '', text)
