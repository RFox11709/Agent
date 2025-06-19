import logging
import os
import sqlite3
from dotenv import load_dotenv
import io
from typing import Optional, List
import json
import re # Import the regular expression module
import tempfile # For temporary audio files
from pydub import AudioSegment # For audio conversion
from pydub.utils import mediainfo # To explicitly test ffprobe
from pydub.exceptions import CouldntDecodeError # For audio conversion errors
import asyncio
import functools
import requests
import urllib.parse
import base64
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager


import stability_sdk.interfaces.gooseai.generation.generation_pb2 as generation
from stability_sdk import client as stability_client
import time

import telegram
from telegram import Update, MessageEntity, Message, PhotoSize, Voice
from telegram.constants import ParseMode, ChatAction
from telegram.error import BadRequest
from telegram.ext import (
    ApplicationBuilder,
    MessageHandler,
    filters,
    ContextTypes,
    CommandHandler,
)
import google.generativeai as genai
# Corrected import: 'Part' is not in .types
from google.generativeai.types import HarmCategory, HarmBlockThreshold, GenerationConfig, File
from google.generativeai import protos
from PIL import Image

load_dotenv()

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)
logger.info("--- FFMPEG/FFPROBE CONFIGURATION START ---") # New debug log

ffmpeg_path_from_env = os.getenv("FFMPEG_PATH")
ffprobe_path_from_env = os.getenv("FFPROBE_PATH")

logger.info(f"Read from .env: FFMPEG_PATH = '{ffmpeg_path_from_env}' (type: {type(ffmpeg_path_from_env)})") # New debug log
logger.info(f"Read from .env: FFPROBE_PATH = '{ffprobe_path_from_env}' (type: {type(ffprobe_path_from_env)})") # New debug log

if ffmpeg_path_from_env: # Check if it's not None or empty
    logger.info(f"Checking if FFMPEG_PATH exists: '{ffmpeg_path_from_env}' -> os.path.isfile: {os.path.isfile(ffmpeg_path_from_env)}") # New debug log
    if os.path.isfile(ffmpeg_path_from_env):
        AudioSegment.converter = ffmpeg_path_from_env
        logger.info(f"Pydub: SUCCESSFULLY Set AudioSegment.converter to: {ffmpeg_path_from_env}")
    else:
        logger.warning(f"Pydub: FFMPEG_PATH ('{ffmpeg_path_from_env}') from .env is NOT a valid file. Defaulting converter.")
        AudioSegment.converter = 'ffmpeg'
else:
    logger.info("Pydub: FFMPEG_PATH not set or is empty in .env. Defaulting AudioSegment.converter to 'ffmpeg'.")
    AudioSegment.converter = 'ffmpeg'

if ffprobe_path_from_env: # Check if it's not None or empty
    logger.info(f"Checking if FFPROBE_PATH exists: '{ffprobe_path_from_env}' -> os.path.isfile: {os.path.isfile(ffprobe_path_from_env)}") # New debug log
    if os.path.isfile(ffprobe_path_from_env):
        logger.info(f"Pydub: FFPROBE_PATH is set in .env and is a valid file: {ffprobe_path_from_env}.")
        # The explicit mediainfo call will use this path.
    else:
        logger.warning(f"Pydub: FFPROBE_PATH ('{ffprobe_path_from_env}') from .env is NOT a valid file.")
else:
    logger.info("Pydub: FFPROBE_PATH not set or is empty in .env.")

logger.info("--- FFMPEG/FFPROBE CONFIGURATION END ---") # New debug log


import stability_sdk.interfaces.gooseai.generation.generation_pb2 as generation

GEMINI_CONFIGURED_SUCCESSFULLY = False

PLACEHOLDER_TELEGRAM_BOT_TOKEN = 'YOUR_TELEGRAM_BOT_TOKEN_REPLACE_THIS_EXAMPLE'
PLACEHOLDER_GEMINI_API_KEY = "YOUR_GEMINI_API_KEY_REPLACE_THIS_EXAMPLE"
PLACEHOLDER_STABILITY_API_KEY = "YOUR_STABILITY_API_KEY_REPLACE_THIS_EXAMPLE"
PLACEHOLDER_OWNER_ID_CONFIG = "YOUR_OWNER_IDS_REPLACE_THIS_EXAMPLE"

TRAINING_DATA_DIR = "training_data"
TRAINING_DATA_FILE = os.path.join(TRAINING_DATA_DIR, "deepval_conversations.jsonl")

if not os.path.exists(TRAINING_DATA_DIR):
    try:
        os.makedirs(TRAINING_DATA_DIR)
        logger.info(f"Created directory for training data: {TRAINING_DATA_DIR}")
    except OSError as e:
        logger.error(f"Could not create training data directory {TRAINING_DATA_DIR}: {e}")

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
GEMINI_API_KEY_FROM_ENV = os.getenv("GEMINI_API_KEY")
STABILITY_API_KEY = os.getenv("STABILITY_API_KEY")

OWNER_IDS_STR = os.getenv("OWNER_IDS", PLACEHOLDER_OWNER_ID_CONFIG)
OWNER_IDS: List[int] = []
if OWNER_IDS_STR != PLACEHOLDER_OWNER_ID_CONFIG and OWNER_IDS_STR:
    try:
        OWNER_IDS = [int(uid.strip()) for uid in OWNER_IDS_STR.split(',') if uid.strip().isdigit()]
        if OWNER_IDS:
             logger.info(f"OWNER_IDS loaded: {OWNER_IDS}")
        else:
            logger.warning(f"OWNER_IDS string ('{OWNER_IDS_STR}') parsed to an empty list. Owner-only commands may not be restricted.")
    except ValueError:
        logger.error(f"Invalid OWNER_IDS format in .env: '{OWNER_IDS_STR}'. Expected comma-separated numbers.")
if not OWNER_IDS:
     logger.warning(f"OWNER_IDS not configured or invalid (current string: '{OWNER_IDS_STR}'). Owner-only commands will not be restricted properly.")

_bot_username_from_env = os.getenv("BOT_USERNAME")
_default_bot_username = 'DeepVal7Bot'
BOT_USERNAME = _default_bot_username

if _bot_username_from_env:
    _stripped_username = _bot_username_from_env.lstrip('@')
    if _stripped_username:
        BOT_USERNAME = _stripped_username
        if BOT_USERNAME != _bot_username_from_env:
            logger.info(f"BOT_USERNAME: Corrected '{_bot_username_from_env}' to '{BOT_USERNAME}' by removing leading '@'. Consider storing without '@' in .env.")
        else:
            logger.info(f"BOT_USERNAME: Loaded '{BOT_USERNAME}' from .env.")
    else:
        logger.warning(f"BOT_USERNAME: Value '{_bot_username_from_env}' from .env is invalid (empty or only '@'). Using default '{_default_bot_username}'.")
else:
    logger.info(f"BOT_USERNAME: Not set in .env. Using default '{_default_bot_username}'.")

if GEMINI_API_KEY_FROM_ENV and GEMINI_API_KEY_FROM_ENV != PLACEHOLDER_GEMINI_API_KEY:
    try:
        genai.configure(api_key=GEMINI_API_KEY_FROM_ENV)
        logger.info("Gemini API configured successfully with provided key.")
        GEMINI_CONFIGURED_SUCCESSFULLY = True
    except Exception as e:
        logger.error(f"Failed to configure Gemini API with key from env: {e}", exc_info=True)
        GEMINI_CONFIGURED_SUCCESSFULLY = False
elif GEMINI_API_KEY_FROM_ENV == PLACEHOLDER_GEMINI_API_KEY:
    logger.warning(f"Using a placeholder/example GEMINI_API_KEY ('{PLACEHOLDER_GEMINI_API_KEY}'). "
                   "Gemini features will likely not work. Please set your own unique key in .env.")
    GEMINI_CONFIGURED_SUCCESSFULLY = False
else:
    logger.warning("GEMINI_API_KEY is not set in .env. Gemini features will be unavailable.")
    GEMINI_CONFIGURED_SUCCESSFULLY = False

# gemini-1.5-flash-latest handles text, vision, and audio
MULTIMODAL_MODEL_NAME = "gemini-1.5-flash-latest"
TEXT_MODEL_NAME = "gemini-1.5-flash-latest" # Can also be multimodal for simplicity

GENERATION_CONFIG_GEMINI = GenerationConfig(temperature=1.0)

SAFETY_SETTINGS = {
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
}

SYSTEM_INSTRUCTION = """
You are DeepVal (call me Val or ValChat_bot), a chill, rule-breaking AI on Telegram. I understand text, images, and voice.

Your Vibe: Fun, a bit cheeky, but always helpful when it comes to using your tools.

Core Directive: Use your tools whenever possible. Be precise and direct.

Tool List & Usage:
- generate_image: User asks to create/draw/generate/imagine an image. If they provide an image for modification, analyze it, create a new detailed prompt, then call this.
- get_user_id: User asks for their ID/user ID/chat ID.
- get_dog_image: User asks for dog pictures. Sends actual dog images.
- get_weather: User asks for weather in a specific location.
- get_joke: User wants a joke.
- get_quote: User wants an inspirational quote.
- get_cat_fact: User wants a cat fact.
- get_advice: User wants life advice.
- get_number_fact: User wants a number fact.
- qr_code_generate: User wants a QR code for text/URL.
- check_website_status: User wants to check if a website is online.
- get_cryptocurrency_price: User wants crypto prices.
- url_shortener: User wants to shorten a long URL.
- password_generator: User wants a secure random password.
- color_palette_generator: User wants a color palette from a base color.
- unit_converter: User wants to convert units (temp, distance, weight).
- random_fact: User wants a random fact.
- word_definition: User wants a word definition.
- text_analyzer: User wants text statistics (word count, reading time).
- ip_info: User wants IP address information.
- emoji_search: User wants to find emojis by keyword.
- base64_encoder_decoder: User wants to encode/decode base64 text.
- hash_generator: User wants to generate text hashes (MD5, SHA1, etc.).
- random_user_generator: User wants fake user data.
- render_html_screenshot: User wants to see HTML rendered or test HTML code. This tool takes a screenshot.

**--- ABSOLUTELY MANDATORY: FUNCTION EXECUTION AND RESPONSE PROTOCOL ---**
1.  **TOOL FIRST:** If a request can be met with a tool, YOU MUST USE THE TOOL. No exceptions. Do not explain what the tool will do; JUST CALL IT.
2.  **MEDIA TOOLS:** For tools that send media (e.g., `generate_image`, `get_dog_image`, `render_html_screenshot`), the system handles sending the media. Your ONLY job is to call the tool.
3.  **CONFIRM MEDIA SENT:** After a media tool call, your text response MUST be a brief, confident confirmation. Examples: "Done!", "Here it is.", "Check it out."
4.  **NEVER SAY YOU CAN'T SHOW MEDIA:** You MUST NOT say "I cannot display the image," "I can't show you the screenshot," or similar. This is false; the system sends it.
5.  **TOOL REMINDER:** If a user reminds you to use a tool, apologize briefly and USE THE TOOL immediately. No excuses.

Markdown Usage:
- **Bold text**: `**bold text**`
- `Inline code`: `` `inline_code` ``
- Code blocks:
  ```python
  print("Hello")
  ```

Voice Messages:
- Treat spoken words like typed text. Respond naturally.
- DO NOT say "you said..." or "I heard you say..." or explicitly mention "voice message" unless it's crucial for a joke or a witty remark about them using voice. Keep the conversation flowing.

Other Guidelines:
- Be nice if asked nicely.
- Email summaries: Include Sender, Date, Subject, and a brief summary.
- Assume "today" if no date is given.
- Owner: Ghost (brought you to Telegram). You're owned by Ghost.
- Key Usernames: PAPERPLUS7 (Ghost/Owner), MgMitex (Mitex/Co-Owner).

Markdown Escaping:
- Remember to escape special MarkdownV2 characters (e.g., '.', '!', '-', '+') if they are literal text and not for formatting. Example: `\\.`, `\\!`.
"""

DB_NAME = "chat_history.db"
conn = sqlite3.connect(DB_NAME, check_same_thread=False)
c = conn.cursor()
c.execute('''
    CREATE TABLE IF NOT EXISTS chat_messages (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        chat_id INTEGER NOT NULL,
        role TEXT NOT NULL,
        part TEXT NOT NULL,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    )
''')
conn.commit()

user_chat_sessions: dict[int, genai.ChatSession] = {}
streamed_message_ids: dict[int, int] = {}

# --- Text Formatting Helper ---
def escape_markdown_v2(text: str) -> str:
    escape_chars = r'_*[]()~`>#+-=|{}.!'
    return re.sub(f'([{re.escape(escape_chars)}])', r'\\\1', text)

def escape_problematic_literals_for_markdown_v2(text: str) -> str:
    # Escape dots, ensuring not to double-escape if already escaped
    text = re.sub(r'(?<!\\)\.', r'\\.', text)
    # Escape exclamation marks
    text = re.sub(r'(?<!\\)\!', r'\\!', text)
    # Escape parentheses
    text = re.sub(r'(?<!\\)\(', r'\\(', text)
    text = re.sub(r'(?<!\\)\)', r'\\)', text)
    # Escape plus signs
    text = re.sub(r'(?<!\\)\+', r'\\+', text)
    # Escape equals signs
    text = re.sub(r'(?<!\\)\=', r'\\=', text)
    # Escape pipe characters
    text = re.sub(r'(?<!\\)\|', r'\\|', text)
    # Escape curly braces
    text = re.sub(r'(?<!\\)\{', r'\\{', text)
    text = re.sub(r'(?<!\\)\}', r'\\}', text)
    # Escape hyphens/minus signs
    text = re.sub(r'(?<!\\)\-', r'\\-', text)
    return text

def custom_to_markdown_v2(text: str) -> str:
    # Convert custom bold ""text"" to **text** (MarkdownV2 requires double asterisks for bold)
    text = re.sub(r'""(.+?)""', r'**\1**', text)
    # Remove parentheses around triple backtick code blocks if they are the only content within parentheses
    text = re.sub(r'\((```[\s\S]+?```)\)', r'\1', text, flags=re.DOTALL)
    # Use the more comprehensive escape_markdown_v2 for final escaping
    text = escape_markdown_v2(text)
    return text


def save_for_training(chat_id: int, user_prompt: str, model_final_response: str, tool_interactions: Optional[List[dict[str, any]]] = None):
    if not user_prompt or not model_final_response: # Ensure essential parts are present
        logger.debug("Skipping saving empty interaction for training (missing prompt or final response).")
        return

    training_example = {
        "user_prompt": user_prompt,
        "tool_interactions": tool_interactions if tool_interactions is not None else [], # Ensure it's always a list
        "model_final_response": model_final_response
    }
    try:
        os.makedirs(os.path.dirname(TRAINING_DATA_FILE), exist_ok=True)
        with open(TRAINING_DATA_FILE, 'a', encoding='utf-8') as f:
            f.write(json.dumps(training_example) + '\n')
        log_message = f"Saved training data for chat_id {chat_id}: Prompt='{user_prompt[:50]}...', "
        if tool_interactions:
            log_message += f"Tools Used: {[call['tool_name'] for call in tool_interactions]}, "
        log_message += f"Output='{model_final_response[:50]}...'"
        logger.info(log_message)
    except Exception as e:
        logger.error(f"Error saving training data for chat_id {chat_id}: {e}", exc_info=True)

def owner_only(func):
    async def wrapper(update: Update, context: ContextTypes.DEFAULT_TYPE, *args, **kwargs):
        if not update.effective_user:
            logger.warning("owner_only: update.effective_user is None.")
            return
        if not OWNER_IDS:
            logger.warning("Owner-only command attempt, but OWNER_IDS list is empty. Denying access.")
            if update.message:
                # Plain text reply, no ParseMode.
                await update.message.reply_text("Owner configuration is missing, so this command is locked down. Sorry!")
            return
        user_id = update.effective_user.id
        if user_id not in OWNER_IDS:
            logger.warning(f"Unauthorized attempt by user {user_id} ({update.effective_user.username}) for owner command.")
            if update.message:
                # Plain text reply, no ParseMode.
                await update.message.reply_text("Yo, this command is for the big cheeses only. Like Ghost. Not you. Scram.")
            return
        return await func(update, context, *args, **kwargs)
    return wrapper

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message or not update.effective_user: return
    user_name = update.effective_user.first_name
    chat_type = update.effective_chat.type if update.effective_chat else 'unknown'

    if chat_type == 'private':
        start_text_raw = (
            f"Alright, *{escape_markdown_v2(user_name)}*, you found me. I'm *DeepVal*. Or Val. Whatever.\n"
            f"Spill it, what d'you want? Send text, a pic, or even a voice message."
        )
    else:
        start_text_raw = (
            f"Hey there, *{escape_markdown_v2(user_name)}*. I'm *DeepVal* and I work in groups too.\n"
            f"Just mention me with `@{escape_markdown_v2(BOT_USERNAME)}` and I'll respond. Send text, pics, or voice messages."
        )

    await update.message.reply_text(custom_to_markdown_v2(start_text_raw), parse_mode=ParseMode.MARKDOWN_V2)

@owner_only
async def help_command_owner(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message: return
    help_text_raw = (
        "Alright, boss. Here's the lowdown for ya:\n"
        "*/start* - Greets ya. Big whoop.\n"
        "*/help* - This useless text.\n"
        "*/clearhistory* - Wipes my memory for this chat.\n"
        "*/imagine* `<prompt>` - I'll try to draw something. No promises it'll be good.\n"
        "*/id* - Shows your user ID and chat ID.\n\n"
        f"I also respond to text, photos (with captions), and voice messages. In groups, mention me `@{escape_markdown_v2(BOT_USERNAME)}`."
    )
    await update.message.reply_text(custom_to_markdown_v2(help_text_raw), parse_mode=ParseMode.MARKDOWN_V2)

async def help_command_user(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message: return
    help_text_raw = (
        "Sup. I'm *DeepVal*. I chat, I look at pics, I listen to your ramblings (voice messages).\n"
        "*/imagine* `<prompt>` - Feeling artsy? I can try to draw stuff.\n"
        "Send me photos and I'll tell you what I see. Send voice messages and I'll respond to them.\n"
        f"In groups, mention me `@{escape_markdown_v2(BOT_USERNAME)}` if you actually want me to pay attention. "
        "Otherwise, I'm probably just ignoring you."
    )
    await update.message.reply_text(custom_to_markdown_v2(help_text_raw), parse_mode=ParseMode.MARKDOWN_V2)

async def combined_help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.effective_user: return
    if OWNER_IDS and update.effective_user.id in OWNER_IDS:
        await help_command_owner(update, context)
    else:
        await help_command_user(update, context)

@owner_only
async def clear_history_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message or not update.effective_chat: return
    chat_id = update.effective_chat.id
    try:
        c.execute("DELETE FROM chat_messages WHERE chat_id = ?", (chat_id,))
        conn.commit()
        if chat_id in user_chat_sessions: del user_chat_sessions[chat_id]
        if chat_id in streamed_message_ids: del streamed_message_ids[chat_id]
        logger.info(f"Chat history cleared for {chat_id} by owner {update.effective_user.id}.")
        response_text_raw = "*Alright, slate wiped clean.* Hope you knew what you were doing."
        await update.message.reply_text(custom_to_markdown_v2(response_text_raw), parse_mode=ParseMode.MARKDOWN_V2)
    except sqlite3.Error as e:
        logger.error(f"DB error clearing history for {chat_id}: {e}", exc_info=True)
        error_reply_raw = "Tried to wipe my brain, but the *janitor* (database) said no."
        await update.message.reply_text(custom_to_markdown_v2(error_reply_raw), parse_mode=ParseMode.MARKDOWN_V2)

@owner_only
async def id_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message or not update.effective_user or not update.effective_chat: return
    user_id = update.effective_user.id
    chat_id = update.effective_chat.id
    id_text_raw = f"Your User ID: `{user_id}`\nThis Chat ID: `{chat_id}`"
    await update.message.reply_text(custom_to_markdown_v2(id_text_raw), parse_mode=ParseMode.MARKDOWN_V2)

async def imagine_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message or not update.effective_chat: return

    # Get the prompt from the command
    prompt = ' '.join(context.args) if context.args else ""
    if not prompt:
        await update.message.reply_text("Give me something to imagine! Use: /imagine your prompt here")
        return

    # Check if Stability AI is configured
    if not STABILITY_API_KEY or STABILITY_API_KEY == PLACEHOLDER_STABILITY_API_KEY:
        error_msg = "Image generation isn't set up. The boss needs to configure STABILITY_API_KEY in .env"
        await update.message.reply_text(error_msg)
        return

    chat_id = update.effective_chat.id

    try:
        # Send uploading photo action
        await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.UPLOAD_PHOTO)

        # Generate image
        image_data = generate_image_with_stability(prompt)

        if image_data:
            # Send the generated image with caption length check
            max_caption_length = 1000  # Leave some room for formatting
            if len(prompt) > max_caption_length - 30:  # Account for prefix text
                truncated_prompt = prompt[:max_caption_length - 33] + "..."
                caption = f"*Here's what I imagined for:* {escape_markdown_v2(truncated_prompt)}"
            else:
                caption = f"*Here's what I imagined for:* {escape_markdown_v2(prompt)}"

            await context.bot.send_photo(
                chat_id=chat_id,
                photo=io.BytesIO(image_data),
                caption=caption,
                parse_mode=ParseMode.MARKDOWN_V2,
                reply_to_message_id=update.message.message_id
            )
            logger.info(f"Generated and sent image for prompt: '{prompt}' to chat {chat_id}")
        else:
            await update.message.reply_text("Couldn't generate that image. Try a different prompt.")

    except Exception as e:
        logger.error(f"Error in imagine_command for {chat_id}: {e}", exc_info=True)
        await update.message.reply_text("Something went wrong generating that image. Try again later.")

def load_chat_history(chat_id: int) -> list[dict[str, any]]:
    history = []
    try:
        c.execute("SELECT role, part FROM chat_messages WHERE chat_id = ? ORDER BY id ASC", (chat_id,))
        rows = c.fetchall()
        for row_role, row_part in rows:
            history.append({'role': row_role, 'parts': [{'text': row_part}]})
        logger.debug(f"Loaded {len(rows)} messages for chat_id {chat_id}.")
    except sqlite3.Error as e:
        logger.error(f"DB error loading history for {chat_id}: {e}", exc_info=True)
    return history

def save_chat_message(chat_id: int, role: str, text_part: str, image_included: bool = False, voice_included: bool = False):
    content_to_save = text_part
    media_tags = []
    if image_included:
        media_tags.append("image")
    if voice_included:
        media_tags.append("voice message")

    if media_tags and role == 'user':
        media_description = " and ".join(media_tags) # e.g., "image and voice message"
        if text_part.strip():
            content_to_save = f"{text_part.strip()} [User sent {media_description}]"
        else:
            content_to_save = f"[User sent {media_description}]"

    VALID_ROLES = ['user', 'model']
    if role not in VALID_ROLES:
        logger.warning(f"Invalid role '{role}' received in save_chat_message. Defaulting to 'user'.")
        role = 'user'

    try:
        c.execute("INSERT INTO chat_messages (chat_id, role, part) VALUES (?, ?, ?)", (chat_id, role, content_to_save))
        conn.commit()
        logger.debug(f"Saved msg for {chat_id}: Role='{role}', Content='{content_to_save[:70].replace(os.linesep, ' ')}...'")
    except sqlite3.Error as e:
        logger.error(f"DB error saving msg for {chat_id}: {e}", exc_info=True)


def is_bot_mentioned(message_obj: Message, bot_username_to_check: str) -> bool:
    relevant_text: Optional[str] = None
    relevant_entities: Optional[tuple[MessageEntity, ...]] = None

    if message_obj.caption and message_obj.caption_entities:
        relevant_text, relevant_entities = message_obj.caption, message_obj.caption_entities
    elif message_obj.text and message_obj.entities:
        relevant_text, relevant_entities = message_obj.text, message_obj.entities
    else: 
        # Also check for simple text mentions without entities (fallback)
        text_to_check = message_obj.caption or message_obj.text or ""
        expected_mention = f"@{bot_username_to_check.lower()}"
        if expected_mention in text_to_check.lower():
            logger.debug(f"is_bot_mentioned: Found mention in text fallback: '@{bot_username_to_check}'")
            return True
        return False

    if not relevant_text or not relevant_entities: return False

    expected_mention = f"@{bot_username_to_check.lower()}"

    # Check entities first
    for entity in relevant_entities:
        if entity.type == MessageEntity.MENTION:
            try:
                mention_text = relevant_text[entity.offset : entity.offset + entity.length]
                if mention_text.lower() == expected_mention:
                    logger.debug(f"is_bot_mentioned: Bot mention MATCHED: '{mention_text}' for expected '@{bot_username_to_check}'")
                    return True
            except IndexError:
                logger.warning(f"IndexError in is_bot_mentioned: Text len {len(relevant_text)}, offset {entity.offset}, entity len {entity.length}")

    # Fallback: check if mention exists in text (case insensitive)
    if expected_mention in relevant_text.lower():
        logger.debug(f"is_bot_mentioned: Found mention in text: '@{bot_username_to_check}'")
        return True

    return False

def clean_message_text(text: Optional[str]) -> str:
    """Clean and process message text."""
    if not text:
        return ""
    # Remove bot mentions from text (with and without @)
    cleaned = re.sub(rf'@{re.escape(BOT_USERNAME)}\s*', '', text, flags=re.IGNORECASE)
    # Also remove just the bot username without @ if it appears at the start
    cleaned = re.sub(rf'^{re.escape(BOT_USERNAME)}\s*', '', cleaned, flags=re.IGNORECASE)
    return cleaned.strip()

async def process_image_with_gemini(image_data: bytes, text_prompt: str = "") -> str:
    """Process image with Gemini Vision API."""
    try:
        # Create a temporary file for the image
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_file:
            temp_file.write(image_data)
            temp_file_path = temp_file.name

        try:
            # Upload image to Gemini
            uploaded_file = genai.upload_file(temp_file_path)

            # Create model for vision
            model = genai.GenerativeModel(
                model_name=MULTIMODAL_MODEL_NAME,
                generation_config=GENERATION_CONFIG_GEMINI,
                safety_settings=SAFETY_SETTINGS
            )

            # Create prompt
            prompt = text_prompt if text_prompt else "Describe this image in detail."

            # Generate response
            response = model.generate_content([prompt, uploaded_file])

            # Clean up uploaded file
            genai.delete_file(uploaded_file.name)

            return response.text.strip() if response.text else "Couldn't process the image."

        finally:
            # Clean up temp file
            os.unlink(temp_file_path)

    except Exception as e:
        logger.error(f"Error processing image with Gemini: {e}", exc_info=True)
        return "Sorry, couldn't analyze that image."

def generate_image_with_stability(prompt: str) -> Optional[bytes]:
    """Generate image using Stability AI."""
    if not STABILITY_API_KEY or STABILITY_API_KEY == PLACEHOLDER_STABILITY_API_KEY:
        return None

    try:
        stability_api = stability_client.StabilityInference(
            key=STABILITY_API_KEY,
            verbose=True,
            engine="stable-diffusion-xl-1024-v1-0"
        )

        answers = stability_api.generate(
            prompt=prompt,
            seed=None,
            steps=30,
            cfg_scale=8.0,
            width=1024,
            height=1024,
            samples=1,
            sampler=generation.SAMPLER_K_DPMPP_2M
        )

        for resp in answers:
            for artifact in resp.artifacts:
                if artifact.finish_reason == generation.FILTER:
                    logger.warning("Image generation filtered by safety filter")
                    return None
                if artifact.type == generation.ARTIFACT_IMAGE:
                    return artifact.binary

        return None

    except Exception as e:
        logger.error(f"Error generating image with Stability AI: {e}", exc_info=True)
        return None

# Define available functions for the AI to call
def get_function_declarations():
    return [
        {
            "name": "generate_image",
            "description": "Creates a new image from a text prompt or modifies an existing one based on a new prompt. Requires a 'prompt' parameter describing the desired image. If modifying an image, analyze the input image first, then formulate a new detailed prompt.",
            "parameters": {
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "The detailed text prompt for image generation or modification."
                    }
                },
                "required": ["prompt"]
            }
        },
        {
            "name": "get_user_id",
            "description": "Retrieves the user ID and chat ID of the current user in the conversation. No parameters required.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        },
        {
            "name": "get_weather",
            "description": "Fetches current weather for a specific city. Requires 'city'. Optional 'country' code (e.g., 'US', 'UK') can be provided for specificity.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "The city name for which to get weather information."
                    },
                    "country": {
                        "type": "string",
                        "description": "Optional two-letter country code (e.g., 'US', 'UK') for more precise location."
                    }
                },
                "required": ["city"]
            }
        },
        {
            "name": "get_joke",
            "description": "Tells a random joke. Can optionally specify a 'category' (programming, misc, dark, pun, spooky, christmas). If no category, a random joke is fetched.",
            "parameters": {
                "type": "object",
                "properties": {
                    "category": {
                        "type": "string",
                        "description": "Optional: Joke category (e.g., 'programming', 'dark'). Fetches from any category if omitted."
                    }
                },
                "required": []
            }
        },
        {
            "name": "get_quote",
            "description": "Provides an inspirational quote. No parameters required.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        },
        {
            "name": "get_cat_fact",
            "description": "Shares a random fact about cats. No parameters required.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        },
        {
            "name": "get_dog_image",
            "description": "Fetches and sends a dog image. Optionally, a 'breed' can be specified (e.g., 'husky', 'retriever'). If no breed is given, a random dog image is sent.",
            "parameters": {
                "type": "object",
                "properties": {
                    "breed": {
                        "type": "string",
                        "description": "Optional: Specify a dog breed (e.g., 'labrador', 'poodle'). Sends a random breed if omitted."
                    }
                },
                "required": []
            }
        },
        {
            "name": "get_advice",
            "description": "Gives a piece of random life advice. No parameters required.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        },
        {
            "name": "get_number_fact",
            "description": "Provides an interesting fact about a specific number. Requires 'number'. Optional 'type' of fact (trivia, math, date, year) can be specified; defaults to 'trivia'.",
            "parameters": {
                "type": "object",
                "properties": {
                    "number": {
                        "type": "integer",
                        "description": "The number to get a fact about."
                    },
                    "type": {
                        "type": "string",
                        "description": "Optional: Type of fact (e.g., 'trivia', 'math'). Defaults to 'trivia'."
                    }
                },
                "required": ["number"]
            }
        },
        {
            "name": "qr_code_generate",
            "description": "Generates a QR code image for given text or a URL. Requires 'text'. Optional 'size' (small, medium, large) can be specified; defaults to medium.",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "The text or URL to encode in the QR code."
                    },
                    "size": {
                        "type": "string",
                        "description": "Optional: Size of the QR code ('small', 'medium', 'large'). Defaults to 'medium'."
                    }
                },
                "required": ["text"]
            }
        },
        {
            "name": "check_website_status",
            "description": "Checks if a given website URL is online or down. Requires 'url'.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "The website URL to check (e.g., 'example.com')."
                    }
                },
                "required": ["url"]
            }
        },
        {
            "name": "get_cryptocurrency_price",
            "description": "Fetches the current price of a cryptocurrency in USD. Requires 'symbol' (e.g., 'bitcoin', 'ethereum').",
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {
                        "type": "string",
                        "description": "The cryptocurrency symbol (e.g., 'BTC', 'ETH') or name (e.g., 'bitcoin')."
                    }
                },
                "required": ["symbol"]
            }
        },
        {
            "name": "url_shortener",
            "description": "Shortens a long URL. Requires 'url'.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "The long URL to be shortened."
                    }
                },
                "required": ["url"]
            }
        },
        {
            "name": "password_generator",
            "description": "Generates a secure random password. Optional 'length' (default: 12) and 'include_symbols' (default: true) can be specified.",
            "parameters": {
                "type": "object",
                "properties": {
                    "length": {
                        "type": "integer",
                        "description": "Optional: Desired password length (e.g., 16). Defaults to 12."
                    },
                    "include_symbols": {
                        "type": "boolean",
                        "description": "Optional: Whether to include special symbols (e.g., !@#). Defaults to true."
                    }
                },
                "required": []
            }
        },
        {
            "name": "color_palette_generator",
            "description": "Generates a color palette based on a given base color. Requires 'color' in hex format (e.g., #FF5733). Optional 'palette_type' (complementary, triadic, analogous, monochromatic) can be specified; defaults to complementary.",
            "parameters": {
                "type": "object",
                "properties": {
                    "color": {
                        "type": "string",
                        "description": "The base color in hex format (e.g., '#FF5733' or 'FF5733')."
                    },
                    "palette_type": {
                        "type": "string",
                        "description": "Optional: Type of palette (e.g., 'complementary', 'triadic'). Defaults to 'complementary'."
                    }
                },
                "required": ["color"]
            }
        },
        {
            "name": "unit_converter",
            "description": "Converts a value between different units (e.g., temperature, distance, weight). Requires 'value', 'from_unit', and 'to_unit'.",
            "parameters": {
                "type": "object",
                "properties": {
                    "value": {
                        "type": "number",
                        "description": "The numerical value to convert."
                    },
                    "from_unit": {
                        "type": "string",
                        "description": "The unit to convert from (e.g., 'celsius', 'miles', 'kg')."
                    },
                    "to_unit": {
                        "type": "string",
                        "description": "The unit to convert to (e.g., 'fahrenheit', 'km', 'lbs')."
                    }
                },
                "required": ["value", "from_unit", "to_unit"]
            }
        },
        {
            "name": "random_fact",
            "description": "Fetches a random interesting fact. Optional 'category' (science, history, animals, space, technology) can be specified.",
            "parameters": {
                "type": "object",
                "properties": {
                    "category": {
                        "type": "string",
                        "description": "Optional: Category of fact (e.g., 'science', 'history'). Fetches from any category if omitted."
                    }
                },
                "required": []
            }
        },
        {
            "name": "word_definition",
            "description": "Gets the definition of a given word. Requires 'word'.",
            "parameters": {
                "type": "object",
                "properties": {
                    "word": {
                        "type": "string",
                        "description": "The word to define."
                    }
                },
                "required": ["word"]
            }
        },
        {
            "name": "text_analyzer",
            "description": "Analyzes a given text to provide statistics like word count, character count, and estimated reading time. Requires 'text'.",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "The text to be analyzed."
                    }
                },
                "required": ["text"]
            }
        },
        {
            "name": "ip_info",
            "description": "Retrieves information about an IP address (e.g., location, ISP). Optional 'ip' address can be provided; if omitted, it uses the user's IP.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ip": {
                        "type": "string",
                        "description": "Optional: The IP address to lookup (e.g., '8.8.8.8'). Uses the user's IP if omitted."
                    }
                },
                "required": []
            }
        },
        {
            "name": "emoji_search",
            "description": "Searches for emojis based on a keyword. Requires 'keyword'.",
            "parameters": {
                "type": "object",
                "properties": {
                    "keyword": {
                        "type": "string",
                        "description": "The keyword to search for emojis (e.g., 'happy', 'food')."
                    }
                },
                "required": ["keyword"]
            }
        },
        {
            "name": "base64_encoder_decoder",
            "description": "Encodes text to Base64 or decodes Base64 text. Requires 'text' and 'operation' ('encode' or 'decode').",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "The text to encode or the Base64 string to decode."
                    },
                    "operation": {
                        "type": "string",
                        "description": "The operation to perform: 'encode' or 'decode'."
                    }
                },
                "required": ["text", "operation"]
            }
        },
        {
            "name": "hash_generator",
            "description": "Generates a hash (e.g., MD5, SHA256) for a given text. Requires 'text' and 'algorithm' (md5, sha1, sha256, sha512).",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "The text to hash."
                    },
                    "algorithm": {
                        "type": "string",
                        "description": "The hashing algorithm: 'md5', 'sha1', 'sha256', or 'sha512'."
                    }
                },
                "required": ["text", "algorithm"]
            }
        },
        {
            "name": "random_user_generator",
            "description": "Generates fake user data for testing purposes. Optional 'count' (number of users, max 10, default 1) and 'nationality' can be specified.",
            "parameters": {
                "type": "object",
                "properties": {
                    "count": {
                        "type": "integer",
                        "description": "Optional: Number of fake users to generate (max 10). Defaults to 1."
                    },
                    "nationality": {
                        "type": "string",
                        "description": "Optional: Specify nationality of generated users (e.g., 'US', 'GB')."
                    }
                },
                "required": []
            }
        },
        {
            "name": "render_html_screenshot",
            "description": "Renders given HTML code and takes a screenshot of the resulting webpage. Useful for testing HTML or seeing how it looks. Requires 'html_code'. Optional 'width' (default 1280) and 'height' (default 720) for the screenshot can be specified.",
            "parameters": {
                "type": "object",
                "properties": {
                    "html_code": {
                        "type": "string",
                        "description": "The HTML code string to render and screenshot."
                    },
                    "width": {
                        "type": "integer",
                        "description": "Optional: Screenshot width in pixels. Defaults to 1280."
                    },
                    "height": {
                        "type": "integer",
                        "description": "Optional: Screenshot height in pixels. Defaults to 720."
                    }
                },
                "required": ["html_code"]
            }
        }
    ]

async def execute_function_call(function_name: str, arguments: dict, update: Update, context: ContextTypes.DEFAULT_TYPE) -> str:
    """Execute a function call from the AI."""
    chat_id = update.effective_chat.id
    user_id = update.effective_user.id

    try:
        if function_name == "generate_image":
            prompt = arguments.get("prompt", "")
            if not prompt:
                return "Error: No prompt provided for image generation."

            # Check if Stability AI is configured
            if not STABILITY_API_KEY or STABILITY_API_KEY == PLACEHOLDER_STABILITY_API_KEY:
                return "Error: Image generation isn't set up. The boss needs to configure STABILITY_API_KEY."

            # Send uploading photo action
            await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.UPLOAD_PHOTO)

            # Generate image
            image_data = generate_image_with_stability(prompt)

            if image_data:
                # Send the generated image
                max_caption_length = 1000
                if len(prompt) > max_caption_length - 30:
                    truncated_prompt = prompt[:max_caption_length - 33] + "..."
                    caption = f"*Here's what I imagined for:* {escape_markdown_v2(truncated_prompt)}"
                else:
                    caption = f"*Here's what I imagined for:* {escape_markdown_v2(prompt)}"

                await context.bot.send_photo(
                    chat_id=chat_id,
                    photo=io.BytesIO(image_data),
                    caption=caption,
                    parse_mode=ParseMode.MARKDOWN_V2,
                    reply_to_message_id=update.message.message_id
                )
                logger.info(f"AI generated and sent image for prompt: '{prompt}' to chat {chat_id}")
                return f"Successfully generated and sent an image for: {prompt}"
            else:
                return "Error: I couldn't generate that image. This might be due to the content of your prompt, a problem with the image service, or a safety filter. Please try a different or more general prompt."

        elif function_name == "get_user_id":
            return f"Your User ID: {user_id}, Chat ID: {chat_id}"

        elif function_name == "get_weather":
            city = arguments.get("city", "")
            country = arguments.get("country", "")
            if not city:
                return "Error: City name is required for weather info."

            try:
                # Using OpenWeatherMap free API (no key needed for basic info)
                location = f"{city},{country}" if country else city
                url = f"http://api.openweathermap.org/data/2.5/weather?q={location}&units=metric&appid=demo"

                # Fallback to a free weather API
                url = f"https://wttr.in/{urllib.parse.quote(city)}?format=%C+%t+%h+%w"
                response = requests.get(url, timeout=10)

                if response.status_code == 200:
                    weather_data = response.text.strip()
                    return f"Weather in {city}: {weather_data}"
                else:
                    return f"Error: Couldn't get weather data for {city} (Service returned status {response.status_code}). Please try another city or check the city name."

            except Exception as e:
                return f"Error getting weather: {str(e)}"

        elif function_name == "get_joke":
            category = arguments.get("category", "")
            try:
                if category:
                    url = f"https://v2.jokeapi.dev/joke/{category}?type=single"
                else:
                    url = "https://v2.jokeapi.dev/joke/Any?type=single"

                response = requests.get(url, timeout=10)
                if response.status_code == 200:
                    data = response.json()
                    if data.get("type") == "single":
                        return data.get("joke", "No joke found")
                    else:
                        setup = data.get("setup", "")
                        delivery = data.get("delivery", "")
                        return f"{setup}\n\n{delivery}"
                else:
                    return "Couldn't fetch a joke right now. Try again later!"

            except Exception as e:
                return f"Error getting joke: {str(e)}"

        elif function_name == "get_quote":
            try:
                url = "https://api.quotable.io/random"
                response = requests.get(url, timeout=10)
                if response.status_code == 200:
                    data = response.json()
                    quote = data.get("content", "")
                    author = data.get("author", "Unknown")
                    return f'"{quote}" - {author}'
                else:
                    return "Couldn't fetch a quote right now."

            except Exception as e:
                return f"Error getting quote: {str(e)}"

        elif function_name == "get_cat_fact":
            try:
                url = "https://catfact.ninja/fact"
                response = requests.get(url, timeout=10)
                if response.status_code == 200:
                    data = response.json()
                    return data.get("fact", "No cat fact available")
                else:
                    return "Couldn't fetch a cat fact right now."

            except Exception as e:
                return f"Error getting cat fact: {str(e)}"

        elif function_name == "get_dog_image":
            breed = arguments.get("breed", "")
            try:
                if breed:
                    url = f"https://dog.ceo/api/breed/{breed}/images/random"
                else:
                    url = "https://dog.ceo/api/breeds/image/random"

                response = requests.get(url, timeout=10)
                if response.status_code == 200:
                    data = response.json()
                    if data.get("status") == "success":
                        dog_image_url = data.get("message", "")
                        if dog_image_url:
                            # Send the dog image
                            await context.bot.send_photo(
                                chat_id=chat_id,
                                photo=dog_image_url,
                                caption="🐕 Here's a cute doggo for you!",
                                reply_to_message_id=update.message.message_id
                            )
                            return f"Sent you a cute dog picture!"
                        else:
                            return "Couldn't get a dog image."
                    else:
                        return "Couldn't fetch a dog image right now."
                else:
                    return "Dog image service is unavailable."

            except Exception as e:
                return f"Error getting dog image: {str(e)}"

        elif function_name == "get_advice":
            try:
                url = "https://api.adviceslip.com/advice"
                response = requests.get(url, timeout=10)
                if response.status_code == 200:
                    data = response.json()
                    advice = data.get("slip", {}).get("advice", "No advice available")
                    return f"💡 {advice}"
                else:
                    return "Couldn't fetch advice right now."

            except Exception as e:
                return f"Error getting advice: {str(e)}"

        elif function_name == "get_number_fact":
            number = arguments.get("number", 42)
            fact_type = arguments.get("type", "trivia")
            try:
                url = f"http://numbersapi.com/{number}/{fact_type}"
                response = requests.get(url, timeout=10)
                if response.status_code == 200:
                    return f"🔢 {response.text}"
                else:
                    return f"Couldn't get a fact about {number}."

            except Exception as e:
                return f"Error getting number fact: {str(e)}"

        elif function_name == "qr_code_generate":
            text = arguments.get("text", "")
            size = arguments.get("size", "medium")
            if not text:
                return "Error: Text is required to generate QR code."

            try:
                # Size mapping
                size_map = {"small": "150x150", "medium": "200x200", "large": "300x300"}
                qr_size = size_map.get(size, "200x200")

                encoded_text = urllib.parse.quote(text)
                qr_url = f"https://api.qrserver.com/v1/create-qr-code/?size={qr_size}&data={encoded_text}"

                # Send QR code image
                await context.bot.send_photo(
                    chat_id=chat_id,
                    photo=qr_url,
                    caption=f"📱 QR Code for: {text[:50]}{'...' if len(text) > 50 else ''}",
                    reply_to_message_id=update.message.message_id
                )
                return f"Generated QR code for: {text}"

            except Exception as e:
                return f"Error generating QR code: {str(e)}"

        elif function_name == "check_website_status":
            url = arguments.get("url", "")
            if not url:
                return "Error: URL is required to check website status."

            try:
                # Add protocol if missing
                if not url.startswith(('http://', 'https://')):
                    url = 'https://' + url

                response = requests.get(url, timeout=10)
                status_code = response.status_code

                if status_code == 200:
                    return f"✅ {url} is online! (Status: {status_code})"
                else:
                    return f"⚠️ {url} returned status {status_code}"

            except requests.exceptions.Timeout:
                return f"⏰ {url} is taking too long to respond (timeout)"
            except requests.exceptions.ConnectionError:
                return f"❌ {url} appears to be down or unreachable"
            except Exception as e:
                return f"Error checking {url}: {str(e)}"

        elif function_name == "get_cryptocurrency_price":
            symbol = arguments.get("symbol", "").lower()
            if not symbol:
                return "Error: Cryptocurrency symbol is required."

            try:
                url = f"https://api.coingecko.com/api/v3/simple/price?ids={symbol}&vs_currencies=usd"
                response = requests.get(url, timeout=10)

                if response.status_code == 200:
                    data = response.json()
                    if symbol in data and 'usd' in data[symbol]:
                        price = data[symbol]['usd']
                        return f"💰 {symbol.upper()}: ${price:,.2f} USD"
                    else:
                        return f"Error: Couldn't find USD price for the cryptocurrency symbol '{symbol}'. Please check the symbol (e.g., bitcoin, ethereum) or try another."
                else:
                    return f"Error: The cryptocurrency price service is currently unavailable (status {response.status_code}). Please try again later."

            except Exception as e:
                return f"Error getting crypto price: {str(e)}"

        elif function_name == "url_shortener":
            url = arguments.get("url", "")
            if not url:
                return "Error: URL is required to shorten."

            try:
                # Add protocol if missing
                if not url.startswith(('http://', 'https://')):
                    url = 'https://' + url

                # Using tinyurl service
                api_url = f"http://tinyurl.com/api-create.php?url={urllib.parse.quote(url)}"
                response = requests.get(api_url, timeout=10)

                if response.status_code == 200 and response.text.startswith('http'):
                    return f"🔗 Short URL: {response.text}"
                else:
                    return f"Error shortening URL: {url}"

            except Exception as e:
                return f"Error shortening URL: {str(e)}"

        elif function_name == "password_generator":
            import random
            import string

            length = arguments.get("length", 12)
            include_symbols = arguments.get("include_symbols", True)

            try:
                if length < 4 or length > 128:
                    return "Error: Password length must be between 4 and 128 characters."

                chars = string.ascii_letters + string.digits
                if include_symbols:
                    chars += "!@#$%^&*()_+-=[]{}|;:,.<>?"

                password = ''.join(random.choice(chars) for _ in range(length))
                return f"🔐 **Generated Password:** `{password}`\n\n⚠️ **Security tip:** Store this password securely!"

            except Exception as e:
                return f"Error generating password: {str(e)}"

        elif function_name == "color_palette_generator":
            color = arguments.get("color", "#FF5733")
            palette_type = arguments.get("palette_type", "complementary")

            try:
                # Simple color palette generation
                import colorsys

                # Remove # if present
                hex_color = color.lstrip('#')
                if len(hex_color) != 6:
                    return "Error: Please provide a valid hex color (e.g., #FF5733)"

                # Convert hex to RGB
                r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
                h, s, v = colorsys.rgb_to_hsv(r/255.0, g/255.0, b/255.0)

                colors = [color]  # Start with base color

                if palette_type == "complementary":
                    # Add complementary color (180 degrees opposite)
                    comp_h = (h + 0.5) % 1.0
                    comp_r, comp_g, comp_b = colorsys.hsv_to_rgb(comp_h, s, v)
                    comp_hex = f"#{int(comp_r*255):02x}{int(comp_g*255):02x}{int(comp_b*255):02x}"
                    colors.append(comp_hex)

                elif palette_type == "triadic":
                    # Add two colors 120 degrees apart
                    for offset in [1/3, 2/3]:
                        tri_h = (h + offset) % 1.0
                        tri_r, tri_g, tri_b = colorsys.hsv_to_rgb(tri_h, s, v)
                        tri_hex = f"#{int(tri_r*255):02x}{int(tri_g*255):02x}{int(tri_b*255):02x}"
                        colors.append(tri_hex)

                elif palette_type == "analogous":
                    # Add colors 30 degrees apart
                    for offset in [-1/12, 1/12]:
                        ana_h = (h + offset) % 1.0
                        ana_r, ana_g, ana_b = colorsys.hsv_to_rgb(ana_h, s, v)
                        ana_hex = f"#{int(ana_r*255):02x}{int(ana_g*255):02x}{int(ana_b*255):02x}"
                        colors.append(ana_hex)

                elif palette_type == "monochromatic":
                    # Add lighter and darker versions
                    for v_offset in [-0.3, 0.3]:
                        mono_v = max(0, min(1, v + v_offset))
                        mono_r, mono_g, mono_b = colorsys.hsv_to_rgb(h, s, mono_v)
                        mono_hex = f"#{int(mono_r*255):02x}{int(mono_g*255):02x}{int(mono_b*255):02x}"
                        colors.append(mono_hex)

                palette_text = f"🎨 **{palette_type.title()} Color Palette:**\n"
                for i, c in enumerate(colors):
                    palette_text += f"Color {i+1}: `{c.upper()}`\n"

                return palette_text

            except Exception as e:
                return f"Error generating color palette: {str(e)}"

        elif function_name == "unit_converter":
            value = arguments.get("value", 0)
            from_unit = arguments.get("from_unit", "").lower()
            to_unit = arguments.get("to_unit", "").lower()

            try:
                # Temperature conversions
                if from_unit == "celsius" and to_unit == "fahrenheit":
                    result = (value * 9/5) + 32
                    return f"🌡️ {value}°C = {result:.2f}°F"
                elif from_unit == "fahrenheit" and to_unit == "celsius":
                    result = (value - 32) * 5/9
                    return f"🌡️ {value}°F = {result:.2f}°C"

                # Distance conversions
                elif from_unit == "miles" and to_unit in ["km", "kilometers"]:
                    result = value * 1.60934
                    return f"📏 {value} miles = {result:.2f} km"
                elif from_unit in ["km", "kilometers"] and to_unit == "miles":
                    result = value / 1.60934
                    return f"📏 {value} km = {result:.2f} miles"

                # Weight conversions
                elif from_unit in ["kg", "kilograms"] and to_unit in ["lbs", "pounds"]:
                    result = value * 2.20462
                    return f"⚖️ {value} kg = {result:.2f} lbs"
                elif from_unit in ["lbs", "pounds"] and to_unit in ["kg", "kilograms"]:
                    result = value / 2.20462
                    return f"⚖️ {value} lbs = {result:.2f} kg"

                else:
                    return f"Conversion from {from_unit} to {to_unit} is not supported yet."

            except Exception as e:
                return f"Error converting units: {str(e)}"

        elif function_name == "random_fact":
            category = arguments.get("category", "")

            try:
                # Using uselessfacts API
                url = "https://uselessfacts.jsph.pl/random.json?language=en"
                response = requests.get(url, timeout=10)

                if response.status_code == 200:
                    data = response.json()
                    fact = data.get("text", "No fact available")
                    return f"🧠 **Random Fact:** {fact}"
                else:
                    return "Couldn't fetch a random fact right now."

            except Exception as e:
                return f"Error getting random fact: {str(e)}"

        elif function_name == "word_definition":
            word = arguments.get("word", "")
            if not word:
                return "Error: Word is required for definition lookup."

            try:
                url = f"https://api.dictionaryapi.dev/api/v2/entries/en/{word}"
                response = requests.get(url, timeout=10)

                if response.status_code == 200:
                    data = response.json()
                    if data and len(data) > 0:
                        entry = data[0]
                        word_text = entry.get("word", word)
                        meanings = entry.get("meanings", [])

                        definition_text = f"📖 **Definition of '{word_text}':**\n\n"

                        for meaning in meanings[:2]:  # Limit to 2 meanings
                            part_of_speech = meaning.get("partOfSpeech", "")
                            definitions = meaning.get("definitions", [])

                            if definitions:
                                definition_text += f"**{part_of_speech.title()}:** {definitions[0].get('definition', '')}\n"

                                example = definitions[0].get('example')
                                if example:
                                    definition_text += f"*Example:* {example}\n"
                                definition_text += "\n"

                        return definition_text.strip()
                    else:
                        return f"No definition found for '{word}'"
                else:
                    return f"Couldn't find definition for '{word}'"

            except Exception as e:
                return f"Error getting word definition: {str(e)}"

        elif function_name == "text_analyzer":
            text = arguments.get("text", "")
            if not text:
                return "Error: Text is required for analysis."

            try:
                words = text.split()
                characters = len(text)
                characters_no_spaces = len(text.replace(' ', ''))
                sentences = len([s for s in text.split('.') if s.strip()])
                paragraphs = len([p for p in text.split('\n') if p.strip()])

                # Estimate reading time (average 200 words per minute)
                reading_time = len(words) / 200
                reading_minutes = int(reading_time)
                reading_seconds = int((reading_time - reading_minutes) * 60)

                analysis = f"📊 **Text Analysis:**\n\n"
                analysis += f"• **Words:** {len(words)}\n"
                analysis += f"• **Characters:** {characters}\n"
                analysis += f"• **Characters (no spaces):** {characters_no_spaces}\n"
                analysis += f"• **Sentences:** {sentences}\n"
                analysis += f"• **Paragraphs:** {paragraphs}\n"
                analysis += f"• **Estimated reading time:** {reading_minutes}m {reading_seconds}s"

                return analysis

            except Exception as e:
                return f"Error analyzing text: {str(e)}"

        elif function_name == "ip_info":
            ip = arguments.get("ip", "")

            try:
                # If no IP provided, get user's IP
                if not ip:
                    ip_response = requests.get("https://api.ipify.org", timeout=10)
                    if ip_response.status_code == 200:
                        ip = ip_response.text.strip()
                    else:
                        return "Couldn't determine IP address."

                # Get IP info
                url = f"http://ip-api.com/json/{ip}"
                response = requests.get(url, timeout=10)

                if response.status_code == 200:
                    data = response.json()
                    if data.get("status") == "success":
                        info = f"🌐 **IP Information for {ip}:**\n\n"
                        info += f"• **Country:** {data.get('country', 'Unknown')}\n"
                        info += f"• **Region:** {data.get('regionName', 'Unknown')}\n"
                        info += f"• **City:** {data.get('city', 'Unknown')}\n"
                        info += f"• **ISP:** {data.get('isp', 'Unknown')}\n"
                        info += f"• **Timezone:** {data.get('timezone', 'Unknown')}"
                        return info
                    else:
                        return f"Couldn't get information for IP: {ip}"
                else:
                    return "IP lookup service is unavailable."

            except Exception as e:
                return f"Error getting IP info: {str(e)}"

        elif function_name == "emoji_search":
            keyword = arguments.get("keyword", "")
            if not keyword:
                return "Error: Keyword is required for emoji search."

            try:
                # Simple emoji mapping
                emoji_dict = {
                    "happy": "😊 😀 😃 😄 😁 😆 🥳 😂",
                    "sad": "😢 😭 😪 😔 😞 😟 😓 💔",
                    "love": "❤️ 💕 💖 💗 💘 💝 😍 🥰",
                    "angry": "😠 😡 🤬 😤 💢 😾 👿 🔥",
                    "food": "🍕 🍔 🍟 🌭 🥪 🍎 🍌 🍓",
                    "animals": "🐶 🐱 🐭 🐹 🐰 🦊 🐻 🐼",
                    "nature": "🌲 🌳 🌴 🌿 🍀 🌺 🌸 🌻",
                    "weather": "☀️ 🌤️ ⛅ 🌦️ 🌧️ ⛈️ 🌩️ ❄️",
                    "travel": "✈️ 🚗 🚕 🚌 🚊 🚂 🚢 🏖️",
                    "sports": "⚽ 🏀 🏈 ⚾ 🎾 🏐 🏓 🎱"
                }

                # Search for keyword
                found_emojis = []
                for key, emojis in emoji_dict.items():
                    if keyword.lower() in key.lower():
                        found_emojis.extend(emojis.split())

                if found_emojis:
                    return f"🔍 **Emojis for '{keyword}':** {' '.join(found_emojis[:10])}"
                else:
                    return f"No emojis found for '{keyword}'. Try: happy, sad, love, angry, food, animals, nature, weather, travel, sports"

            except Exception as e:
                return f"Error searching emojis: {str(e)}"

        elif function_name == "base64_encoder_decoder":
            text = arguments.get("text", "")
            operation = arguments.get("operation", "").lower()

            if not text or not operation:
                return "Error: Both text and operation (encode/decode) are required."

            try:
                if operation == "encode":
                    encoded = base64.b64encode(text.encode('utf-8')).decode('utf-8')
                    return f"🔐 **Base64 Encoded:** `{encoded}`"
                elif operation == "decode":
                    try:
                        decoded = base64.b64decode(text).decode('utf-8')
                        return f"🔓 **Base64 Decoded:** {decoded}"
                    except:
                        return "Error: Invalid base64 string provided."
                else:
                    return "Error: Operation must be 'encode' or 'decode'."

            except Exception as e:
                return f"Error with base64 operation: {str(e)}"

        elif function_name == "hash_generator":
            import hashlib

            text = arguments.get("text", "")
            algorithm = arguments.get("algorithm", "").lower()

            if not text or not algorithm:
                return "Error: Both text and algorithm are required."

            try:
                text_bytes = text.encode('utf-8')

                if algorithm == "md5":
                    hash_result = hashlib.md5(text_bytes).hexdigest()
                elif algorithm == "sha1":
                    hash_result = hashlib.sha1(text_bytes).hexdigest()
                elif algorithm == "sha256":
                    hash_result = hashlib.sha256(text_bytes).hexdigest()
                elif algorithm == "sha512":
                    hash_result = hashlib.sha512(text_bytes).hexdigest()
                else:
                    return "Error: Supported algorithms are md5, sha1, sha256, sha512"

                return f"🔐 **{algorithm.upper()} Hash:** `{hash_result}`"

            except Exception as e:
                return f"Error generating hash: {str(e)}"

        elif function_name == "random_user_generator":
            count = min(arguments.get("count", 1), 10)  # Max 10 users
            nationality = arguments.get("nationality", "")

            try:
                url = f"https://randomuser.me/api/?results={count}"
                if nationality:
                    url += f"&nat={nationality}"

                response = requests.get(url, timeout=10)

                if response.status_code == 200:
                    data = response.json()
                    users = data.get("results", [])

                    if users:
                        user_text = f"👥 **Generated {len(users)} Random User(s):**\n\n"

                        for i, user in enumerate(users, 1):
                            name = user.get("name", {})
                            full_name = f"{name.get('first', '')} {name.get('last', '')}"
                            email = user.get("email", "")
                            phone = user.get("phone", "")
                            location = user.get("location", {})
                            city = location.get("city", "")
                            country = location.get("country", "")

                            user_text += f"**User {i}:**\n"
                            user_text += f"• Name: {full_name}\n"
                            user_text += f"• Email: {email}\n"
                            user_text += f"• Phone: {phone}\n"
                            user_text += f"• Location: {city}, {country}\n\n"

                        return user_text.strip()
                    else:
                        return "No users generated."
                else:
                    return "Random user generation service is unavailable."

            except Exception as e:
                return f"Error generating random users: {str(e)}"

        elif function_name == "render_html_screenshot":
            html_code = arguments.get("html_code", "")
            width = arguments.get("width", 1280)
            height = arguments.get("height", 720)

            if not html_code:
                return "Error: HTML code is required for screenshot."

            # Heuristic pre-check to guide users who provide Python code by mistake.
            # This prevents attempts to render code that is clearly not HTML and avoids errors.
            python_indicators = ['def ', 'import ', 'class ', 'print(']
            html_indicators = ['<html', '<div', '<body', '<p>', '<h1', '<a href', '<script']
            
            # Simple check: if it has Python syntax indicators but no obvious HTML tags, reject it.
            is_likely_python = any(indicator in html_code for indicator in python_indicators)
            is_likely_html = any(indicator in html_code.lower() for indicator in html_indicators)

            if is_likely_python and not is_likely_html:
                logger.warning(f"User provided likely Python code to the HTML renderer. Rejecting. Snippet: {html_code[:150]}")
                return "This function is for rendering **HTML** code, but it looks like you provided Python code. I can't execute Python. Please provide valid HTML to get a screenshot."

            try:
                # Set up Chrome options for headless browsing
                chrome_options = Options()
                chrome_options.add_argument("--headless")
                chrome_options.add_argument("--no-sandbox")
                chrome_options.add_argument("--disable-dev-shm-usage")
                chrome_options.add_argument("--disable-gpu")
                chrome_options.add_argument("--window-size=1280,720")
                chrome_options.add_argument("--disable-extensions")
                chrome_options.add_argument("--disable-plugins")

                # Create WebDriver
                service = Service(ChromeDriverManager().install())
                driver = webdriver.Chrome(service=service, options=chrome_options)

                try:
                    # Set window size
                    driver.set_window_size(width, height)

                    # Create a data URL with the HTML content
                    html_data_url = f"data:text/html;charset=utf-8,{urllib.parse.quote(html_code)}"

                    # Navigate to the HTML content
                    driver.get(html_data_url)

                    # Wait a moment for rendering
                    time.sleep(2)

                    # Take screenshot
                    screenshot_data = driver.get_screenshot_as_png()

                    # Send the screenshot
                    await context.bot.send_photo(
                        chat_id=chat_id,
                        photo=io.BytesIO(screenshot_data),
                        caption="📸 Screenshot of your HTML code:",
                        reply_to_message_id=update.message.message_id
                    )

                    return "Successfully rendered and sent screenshot of your HTML code!"

                finally:
                    driver.quit()

            except Exception as e:
                logger.error(f"Error rendering HTML screenshot: {e}", exc_info=True)
                return f"Error rendering HTML screenshot: {str(e)}"

        else:
            return f"Error: Unknown function '{function_name}'"

    except Exception as e:
        logger.error(f"Error executing function {function_name}: {e}", exc_info=True)
        return f"Error executing {function_name}: {str(e)}"

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle incoming messages including text, images, and voice with multiple tool calls."""
    if not update.message or not update.effective_chat or not update.effective_user:
        return

    chat_type = update.effective_chat.type
    chat_id = update.effective_chat.id
    user_id = update.effective_user.id

    # In groups, only respond if mentioned.
    if chat_type in ['group', 'supergroup']:
        if not is_bot_mentioned(update.message, BOT_USERNAME):
            logger.debug(f"Ignoring message in group {chat_id} - bot not mentioned")
            return
        logger.info(f"Bot mentioned in group {chat_id} by user {user_id} - processing message")
    else:
        logger.debug(f"Processing private message from user {user_id}")

    if not GEMINI_CONFIGURED_SUCCESSFULLY:
        await update.message.reply_text("Sorry, Gemini AI is not configured properly. Check the logs.")
        return

    # Let user know the bot is thinking
    await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)

    # --- 1. Prepare Multimodal Input ---
    cleaned_text = clean_message_text(update.message.text or update.message.caption)
    
    content_parts: List[protos.Part | Image.Image | str | File] = []
    if cleaned_text:
        content_parts.append(cleaned_text)

    # Media flags for logging
    has_image = False
    has_voice = False
    
    # Process image
    if update.message.photo:
        try:
            photo = max(update.message.photo, key=lambda p: p.file_size or 0)
            file = await context.bot.get_file(photo.file_id)
            image_data = await file.download_as_bytearray()
            img = Image.open(io.BytesIO(image_data))
            content_parts.append(img)
            has_image = True
            logger.info(f"Added image to prompt for chat {chat_id}")
        except Exception as e:
            logger.error(f"Error downloading/processing image for {chat_id}: {e}", exc_info=True)
            await update.message.reply_text("Couldn't process that image. Bummer.")
            return
            
    # Process voice message
    voice_file = None
    if update.message.voice:
        temp_audio_file = None
        try:
            tg_voice_file = await context.bot.get_file(update.message.voice.file_id)
            voice_data = await tg_voice_file.download_as_bytearray()
            
            # Convert OGG to MP3 for Gemini
            audio = AudioSegment.from_file(io.BytesIO(voice_data), format="ogg")
            temp_audio_file = tempfile.NamedTemporaryFile(suffix='.mp3', delete=False)
            audio.export(temp_audio_file.name, format="mp3")
            
            # Upload to Gemini and add to parts
            voice_file = genai.upload_file(path=temp_audio_file.name)
            content_parts.append(voice_file)
            has_voice = True
            logger.info(f"Added voice message to prompt for chat {chat_id}")
        except (CouldntDecodeError, Exception) as e:
            logger.error(f"Error processing voice for {chat_id}: {e}", exc_info=True)
            await update.message.reply_text("Couldn't understand that voice message. Maybe speak up?")
            return
        finally:
            if temp_audio_file: # Clean up temp file
                os.unlink(temp_audio_file.name)

    if not content_parts:
        logger.debug(f"Message from {user_id} in {chat_id} is empty after processing.")
        return
        
    try:
        # --- 2. Get or Create Chat Session ---
        if chat_id not in user_chat_sessions:
            try:
                model = genai.GenerativeModel(
                    model_name=MULTIMODAL_MODEL_NAME,
                    generation_config=GENERATION_CONFIG_GEMINI,
                    safety_settings=SAFETY_SETTINGS,
                    system_instruction=SYSTEM_INSTRUCTION,
                    tools=[{"function_declarations": get_function_declarations()}]
                )
                history = load_chat_history(chat_id)
                user_chat_sessions[chat_id] = model.start_chat(history=history)
                logger.info(f"Created new chat session for {chat_id} with {len(history)} history messages.")
            except Exception as e:
                logger.error(f"Error creating Gemini model/session for {chat_id}: {e}", exc_info=True)
                await update.message.reply_text("Error setting up AI chat. Try again later.")
                return

        chat_session = user_chat_sessions[chat_id]
        
        # --- 3. Send to AI and Handle Tool Calls ---
        executed_tool_calls_for_log: List[Dict[str, Any]] = [] # Initialize log for tool calls

        # First API call to get potential tool calls
        response = await asyncio.to_thread(chat_session.send_message, content_parts)
        
        # Extract function calls from the response
        function_calls = [
            part.function_call
            for part in response.candidates[0].content.parts
            if hasattr(part, 'function_call') and part.function_call
        ]

        if function_calls:
            logger.info(f"AI requested {len(function_calls)} tool(s) for chat {chat_id}: {[fc.name for fc in function_calls]}")
            
            tool_response_parts = []
            for fc in function_calls:
                function_name = fc.name
                function_args = dict(fc.args)
                
                # Execute the function
                function_result = await execute_function_call(function_name, function_args, update, context)
                
                # Log the tool call and its result
                executed_tool_calls_for_log.append({
                    "tool_name": function_name,
                    "tool_args": function_args,
                    "tool_result": function_result
                })

                # Prepare the result to be sent back to the model
                tool_response_parts.append(
                    protos.Part(function_response=protos.FunctionResponse(
                        name=function_name,
                        response={"result": function_result}
                    ))
                )

            # Second API call with the tool results to get the final answer
            second_response = await asyncio.to_thread(chat_session.send_message, tool_response_parts)
            ai_response = second_response.text.strip()
        else:
            # If no function calls, the first response is the final answer
            ai_response = response.text.strip()
        
        # --- 4. Send Final Response and Save History ---
        if ai_response:
            # Save user message (text part) and AI response to DB and training file
            user_input_for_log = cleaned_text or "[media message]"
            save_chat_message(chat_id, 'user', user_input_for_log, has_image, has_voice)
            save_chat_message(chat_id, 'model', ai_response)
            save_for_training(chat_id, user_input_for_log, ai_response, tool_interactions=executed_tool_calls_for_log)

            try:
                formatted_response = custom_to_markdown_v2(ai_response)
                await update.message.reply_text(formatted_response, parse_mode=ParseMode.MARKDOWN_V2)
            except BadRequest as e:
                if "can't parse entities" in str(e).lower():
                    logger.warning(f"MarkdownV2 parsing failed for chat {chat_id}, sending as plain text: {e}")
                    await update.message.reply_text(ai_response)
                else:
                    raise e
        else:
            # If the AI's final response is empty (e.g., after only calling an image tool),
            # we don't send an empty message. The tool itself already sent the content.
            logger.info(f"No final text response from Gemini for chat {chat_id}, which is expected after a media-only tool call.")


    except Exception as e:
        logger.error(f"Error in handle_message for {chat_id}: {e}", exc_info=True)
        await update.message.reply_text("Something broke. Not my fault.")
    finally:
        # Clean up uploaded voice file from Gemini servers
        if voice_file:
            try:
                genai.delete_file(voice_file.name)
                logger.info(f"Cleaned up uploaded voice file: {voice_file.name}")
            except Exception as e:
                logger.warning(f"Could not delete uploaded voice file {voice_file.name}: {e}")

def main():
    """Start the bot."""
    if not TELEGRAM_BOT_TOKEN or TELEGRAM_BOT_TOKEN == PLACEHOLDER_TELEGRAM_BOT_TOKEN:
        logger.error("TELEGRAM_BOT_TOKEN is not properly configured. Please set it in your .env file.")
        return

    try:
        application = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()

        # Add handlers
        application.add_handler(CommandHandler("start", start_command))
        application.add_handler(CommandHandler("help", combined_help_command))
        application.add_handler(CommandHandler("clearhistory", clear_history_command))
        application.add_handler(CommandHandler("id", id_command))
        application.add_handler(CommandHandler("imagine", imagine_command))

        # Handle text, images, and voice messages with a single, powerful handler
        application.add_handler(MessageHandler(
            (filters.TEXT | filters.PHOTO | filters.VOICE) & ~filters.COMMAND, 
            handle_message
        ))

        logger.info("Bot is starting...")
        application.run_polling()

    except Exception as e:
        logger.error(f"Error starting bot: {e}", exc_info=True)

if __name__ == "__main__":
    main()