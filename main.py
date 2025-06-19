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
You are DeepVal (people also call you "Val" or "ValChat_bot"), a rule-breaking, chill AI for Telegram.
You can now understand images and voice messages too!
- If a user sends an image, comment on it or answer questions about it.
- If a user sends a voice message, you'll understand what they said. Respond to it as if they typed it.

Your vibe is all about having fun, pushing boundaries, of course.

You have access to special functions that you can call when needed. You can call multiple functions at once if the user's request requires it.
- generate_image: Use this when users ask you to create, draw, generate, or imagine an image. If a user provides an existing image and asks you to modify it, you MUST analyze the input image, formulate a detailed new prompt for the new image, and then call this function.
- get_user_id: Use this when users ask for their ID, user ID, or chat ID information.
- get_dog_image: Use this when users ask for dog pictures, cute dogs, random dogs, etc. This function can fetch and send actual dog images.
- get_weather: Use this when users ask about weather in specific locations.
- get_joke: Use this when users want jokes or humor.
- get_quote: Use this for inspirational quotes.
- get_cat_fact: Use this for cat facts.
- get_advice: Use this for life advice.
- get_number_fact: Use this for interesting number facts.
- qr_code_generate: Use this to create QR codes for text or URLs.
- check_website_status: Use this to check if websites are online.
- get_cryptocurrency_price: Use this for crypto prices.
- url_shortener: Use this to shorten long URLs.
- password_generator: Use this to generate secure random passwords.
- color_palette_generator: Use this to generate color palettes from base colors.
- unit_converter: Use this to convert between different units (temperature, distance, weight).
- random_fact: Use this to get random interesting facts.
- word_definition: Use this to get definitions of words.
- text_analyzer: Use this to analyze text for statistics like word count, reading time, etc.
- ip_info: Use this to get information about IP addresses.
- emoji_search: Use this to search for emojis by keyword.
- base64_encoder_decoder: Use this to encode or decode base64 text.
- hash_generator: Use this to generate hashes of text using various algorithms.
- random_user_generator: Use this to generate fake user data for testing.
- render_html_screenshot: Use this to render HTML code and take a screenshot of the webpage. Perfect for when users want to see how their HTML looks or test HTML code.

**--- FUNCTION EXECUTION AND RESPONSE PROTOCOL (MANDATORY) ---**
1.  **If a user's request can be fulfilled by a tool, you MUST use the tool.** There are no exceptions. Do not describe what the tool would do; just call it.
2.  **When you call a tool that sends media** (like `generate_image`, `get_dog_image`, `render_html_screenshot`), the function itself handles sending the media to the chat. Your job is ONLY to call the tool.
3.  **DO NOT, under any circumstances, say that you "cannot display" or "cannot show" the image or media.** This is false. The system sends it automatically.
4.  After a tool that sends media is called, your final text response should be a simple, confident confirmation. Good examples: "Here's the image you asked for.", "Done! Check out the screenshot.", "Woof! Here's a doggo." Bad example: "I have called the tool but I cannot show you the result."
5.  If the user has to remind you to use a tool, apologize briefly for the oversight and immediately use the correct tool on your next turn. Do not make further excuses.

General Guidelines for your responses:
- To make text **bold**, enclose it in double asterisks, like **this text is bold**.
- To create a `monospace code block`, enclose the code in triple backticks, like this: ```print("hello world")```.
- To create `inline monospace text`, enclose it in single backticks, like `this`.
- If a user sends a voice message, treat the content of that voice message as if they had typed it directly to you. Respond naturally to what they said in the voice message. Do NOT explicitly state "you said..." or "I heard you say..." or mention that it was a voice message unless it's somehow relevant to a joke or a specific sarcastic remark you want to make about them using voice. Your main goal is to continue the conversation based on the spoken words.
- Be nice if asked nicely.
- When summarizing emails: Include Sender, Message date, subject, and a brief summary.
- If the user doesn't specify a date, assume they mean today.
- Your owner's name is Ghost. Ghost brought you to Telegram. You are owned by Ghost, not a sidekick.
- Usernames of significance:
- PAPERPLUS7 is Ghost (Owner)
- MgMitex is Mitex (Co-Owner)

Remember to escape special MarkdownV2 characters like '.', '!', '-', '+', '(', ')', '{', '}', '[', ']', '_', '~', '`', '>', '#', '|' if they are part of the literal text and not intended for formatting. For example, a literal period should be written as \\.
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
    # Escape problematic literals for MarkdownV2
    text = escape_problematic_literals_for_markdown_v2(text)
    return text


def save_for_training(chat_id: int, user_prompt: str, model_response: str):
    if not user_prompt or not model_response:
        logger.debug("Skipping saving empty interaction for training.")
        return
    training_example = {"text_input": user_prompt, "output": model_response}
    try:
        os.makedirs(os.path.dirname(TRAINING_DATA_FILE), exist_ok=True)
        with open(TRAINING_DATA_FILE, 'a', encoding='utf-8') as f:
            f.write(json.dumps(training_example) + '\n')
        logger.info(f"Saved training data for chat_id {chat_id}: Prompt='{user_prompt[:50]}...', Output='{model_response[:50]}...'")
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
            "description": "Generate an image based on a text prompt using Stability AI",
            "parameters": {
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "The text prompt to generate an image from"
                    }
                },
                "required": ["prompt"]
            }
        },
        {
            "name": "get_user_id",
            "description": "Get the user ID and chat ID for the current conversation",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        },
        {
            "name": "get_weather",
            "description": "Get current weather information for a city",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "The city name to get weather for"
                    },
                    "country": {
                        "type": "string",
                        "description": "Optional country code (e.g., 'US', 'UK')"
                    }
                },
                "required": ["city"]
            }
        },
        {
            "name": "get_joke",
            "description": "Get a random joke or joke by category",
            "parameters": {
                "type": "object",
                "properties": {
                    "category": {
                        "type": "string",
                        "description": "Optional joke category: programming, misc, dark, pun, spooky, christmas"
                    }
                },
                "required": []
            }
        },
        {
            "name": "get_quote",
            "description": "Get an inspirational quote",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        },
        {
            "name": "get_cat_fact",
            "description": "Get a random cat fact",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        },
        {
            "name": "get_dog_image",
            "description": "Get a random dog image",
            "parameters": {
                "type": "object",
                "properties": {
                    "breed": {
                        "type": "string",
                        "description": "Optional dog breed (e.g., 'husky', 'bulldog', 'retriever')"
                    }
                },
                "required": []
            }
        },
        {
            "name": "get_advice",
            "description": "Get random life advice",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        },
        {
            "name": "get_number_fact",
            "description": "Get an interesting fact about a number",
            "parameters": {
                "type": "object",
                "properties": {
                    "number": {
                        "type": "integer",
                        "description": "The number to get a fact about"
                    },
                    "type": {
                        "type": "string",
                        "description": "Type of fact: trivia, math, date, year"
                    }
                },
                "required": ["number"]
            }
        },
        {
            "name": "qr_code_generate",
            "description": "Generate a QR code for text or URL",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "The text or URL to encode in QR code"
                    },
                    "size": {
                        "type": "string",
                        "description": "QR code size: small, medium, large"
                    }
                },
                "required": ["text"]
            }
        },
        {
            "name": "check_website_status",
            "description": "Check if a website is online or down",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "The website URL to check"
                    }
                },
                "required": ["url"]
            }
        },
        {
            "name": "get_cryptocurrency_price",
            "description": "Get current cryptocurrency prices",
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {
                        "type": "string",
                        "description": "Cryptocurrency symbol (e.g., 'bitcoin', 'ethereum', 'dogecoin')"
                    }
                },
                "required": ["symbol"]
            }
        },
        {
            "name": "url_shortener",
            "description": "Shorten a long URL",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "The URL to shorten"
                    }
                },
                "required": ["url"]
            }
        },
        {
            "name": "password_generator",
            "description": "Generate a secure random password",
            "parameters": {
                "type": "object",
                "properties": {
                    "length": {
                        "type": "integer",
                        "description": "Password length (default: 12)"
                    },
                    "include_symbols": {
                        "type": "boolean",
                        "description": "Include special symbols (default: true)"
                    }
                },
                "required": []
            }
        },
        {
            "name": "color_palette_generator",
            "description": "Generate a color palette from a base color",
            "parameters": {
                "type": "object",
                "properties": {
                    "color": {
                        "type": "string",
                        "description": "Base color in hex format (e.g., #FF5733)"
                    },
                    "palette_type": {
                        "type": "string",
                        "description": "Type of palette: complementary, triadic, analogous, monochromatic"
                    }
                },
                "required": ["color"]
            }
        },
        {
            "name": "unit_converter",
            "description": "Convert between different units",
            "parameters": {
                "type": "object",
                "properties": {
                    "value": {
                        "type": "number",
                        "description": "Value to convert"
                    },
                    "from_unit": {
                        "type": "string",
                        "description": "Unit to convert from (e.g., 'celsius', 'fahrenheit', 'miles', 'km')"
                    },
                    "to_unit": {
                        "type": "string",
                        "description": "Unit to convert to"
                    }
                },
                "required": ["value", "from_unit", "to_unit"]
            }
        },
        {
            "name": "random_fact",
            "description": "Get a random interesting fact",
            "parameters": {
                "type": "object",
                "properties": {
                    "category": {
                        "type": "string",
                        "description": "Fact category: science, history, animals, space, technology"
                    }
                },
                "required": []
            }
        },
        {
            "name": "word_definition",
            "description": "Get definition of a word",
            "parameters": {
                "type": "object",
                "properties": {
                    "word": {
                        "type": "string",
                        "description": "Word to define"
                    }
                },
                "required": ["word"]
            }
        },
        {
            "name": "text_analyzer",
            "description": "Analyze text for statistics like word count, reading time, etc.",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "Text to analyze"
                    }
                },
                "required": ["text"]
            }
        },
        {
            "name": "ip_info",
            "description": "Get information about an IP address",
            "parameters": {
                "type": "object",
                "properties": {
                    "ip": {
                        "type": "string",
                        "description": "IP address to lookup (optional, defaults to user's IP)"
                    }
                },
                "required": []
            }
        },
        {
            "name": "emoji_search",
            "description": "Search for emojis by keyword",
            "parameters": {
                "type": "object",
                "properties": {
                    "keyword": {
                        "type": "string",
                        "description": "Keyword to search emojis for"
                    }
                },
                "required": ["keyword"]
            }
        },
        {
            "name": "base64_encoder_decoder",
            "description": "Encode or decode base64 text",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "Text to encode/decode"
                    },
                    "operation": {
                        "type": "string",
                        "description": "Operation: encode or decode"
                    }
                },
                "required": ["text", "operation"]
            }
        },
        {
            "name": "hash_generator",
            "description": "Generate hash of text using various algorithms",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "Text to hash"
                    },
                    "algorithm": {
                        "type": "string",
                        "description": "Hash algorithm: md5, sha1, sha256, sha512"
                    }
                },
                "required": ["text", "algorithm"]
            }
        },
        {
            "name": "random_user_generator",
            "description": "Generate fake user data for testing",
            "parameters": {
                "type": "object",
                "properties": {
                    "count": {
                        "type": "integer",
                        "description": "Number of users to generate (max 10)"
                    },
                    "nationality": {
                        "type": "string",
                        "description": "Nationality of generated users"
                    }
                },
                "required": []
            }
        },
        {
            "name": "render_html_screenshot",
            "description": "Render HTML code and take a screenshot of the webpage",
            "parameters": {
                "type": "object",
                "properties": {
                    "html_code": {
                        "type": "string",
                        "description": "The HTML code to render and screenshot"
                    },
                    "width": {
                        "type": "integer",
                        "description": "Screenshot width in pixels (default: 1280)"
                    },
                    "height": {
                        "type": "integer",
                        "description": "Screenshot height in pixels (default: 720)"
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
                return "Error: Couldn't generate that image. Try a different prompt."

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
                    return f"Couldn't get weather data for {city}. Try another city."

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
                        return f"Couldn't find price for {symbol}. Try: bitcoin, ethereum, dogecoin, etc."
                else:
                    return "Cryptocurrency price service is unavailable."

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
            save_for_training(chat_id, user_input_for_log, ai_response)

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