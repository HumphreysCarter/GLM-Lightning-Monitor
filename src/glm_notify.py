import time
import json
import smtplib
import logging
import requests
from pathlib import Path
from email.message import EmailMessage
from email.utils import formataddr

from .settings import NOTIFY_EMAILS, EMAIL_SERVER, EMAIL_PORT, EMAIL_USER, EMAIL_PASS, WATCH_LATITUDE, WATCH_LONGITUDE, \
    NOTIFY_WITHIN_MILES, CHECK_LAST_MINUTES, API_PORT

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def send_email(subject, body):
    msg = EmailMessage()
    msg['From'] = formataddr(('GLM Notification', EMAIL_USER))
    msg['To'] = NOTIFY_EMAILS
    msg['Subject'] = subject
    msg.set_content(body)

    try:
        with smtplib.SMTP_SSL(EMAIL_SERVER, EMAIL_PORT) as server:
            server.login(EMAIL_USER, EMAIL_PASS)
            server.send_message(msg)
        logger.info('Email sent!')
    except Exception as e:
        logger.warning(f"Failed to send email: {e}")


def fetch_nearby_stats(lat, lon, miles=50, minutes=30, use_cache=True, timeout=10):
    params = {
        'lat': lat,
        'lon': lon,
        'miles': miles,
        'minutes': minutes,
        'use_cache': str(use_cache).lower(),
    }
    resp = requests.get(f'http://localhost:{API_PORT}/stats/nearby', params=params, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def main():
    last_check_file = Path('/app', 'data', 'last_check.json')

    # Check for lightning
    current_stats = fetch_nearby_stats(WATCH_LATITUDE, WATCH_LONGITUDE, miles=NOTIFY_WITHIN_MILES,
                                       minutes=CHECK_LAST_MINUTES)

    if current_stats['count'] == 0:
        logger.info('No GLM events within range of location')
        with open(last_check_file, 'w') as f:
            json.dump(current_stats, f)
        return

    # Check previous
    if last_check_file.is_file():
        with open(last_check_file, 'r') as f:
            prev_stats = json.load(f)
    else:
        prev_stats = None

    distance_change = ''
    if prev_stats and prev_stats['count'] > 0:
        current_distance = current_stats['avg_distance_miles']
        previous_distance = prev_stats['avg_distance_miles']

        if current_distance < previous_distance:
            distance_change = ' and decreasing'
        else:
            distance_change = ' and increasing'

    logger.info('Sending notification')
    send_email('⚡ Lightning Detected ⚡',
               f'{current_stats["count"]} flashes within the last {CHECK_LAST_MINUTES} minutes. Average distance is {current_stats["avg_distance_miles"]:.0f} miles{distance_change}.')

    with open(last_check_file, 'w') as f:
        json.dump(current_stats, f)

if __name__ == "__main__":
    logger.info(f'Waiting 30 seconds to allow for data ingest...')
    time.sleep(30)
    logger.info(f'Starting GLM watch...')

    while True:
        main()
        logger.info(f'Waiting 300 seconds before checking for GLM events')
        time.sleep(300)
