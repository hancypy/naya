import logging
import asyncio
from telegram import Bot
from telegram.constants import ParseMode
from telegram.error import TelegramError

# Configure logging
logging.basicConfig(
    filename='telegram_test.log',
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.DEBUG  # Set to DEBUG to capture all logs
)

async def send_test_telegram_alert():
    try:
        bot_token = '7878474153:AAGxb39AIdALbRYobjkCiJXQdbyb4ReC6Uw'  # Replace with your actual bot token
        chat_id = '-4625746737'      # Replace with your actual chat ID
        bot = Bot(token=bot_token)
        message = "🚀 **Test Alert** 🚀\nbittal bohot badha gandu hai."
        await bot.send_message(chat_id=chat_id, text=message, parse_mode=ParseMode.MARKDOWN)
        logging.info("Test Telegram alert sent successfully.")
    except TelegramError as e:
        logging.error(f"Failed to send test Telegram alert: {e}")
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    asyncio.run(send_test_telegram_alert())
