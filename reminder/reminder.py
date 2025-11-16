import os
import requests
import schedule
import time
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# =========================
# CONFIGURATION
# =========================

from dotenv import load_dotenv
load_dotenv()

API_KEY = os.getenv("API_KEY")
SENDER_EMAIL = os.getenv("SENDER_EMAIL")
SENDER_PASSWORD = os.getenv("SENDER_PASSWORD")
RECEIVER_EMAIL = os.getenv("RECEIVER_EMAIL")

HOT_TEMP_THRESHOLD = 30  # °C
HOT_INTERVAL = 1         # hours between reminders if hot
NORMAL_INTERVAL = 3      # hours if not hot

API_KEY = "c0937e198d079cd35819720f4e44ce32"
CITY = "Tunis"

SENDER_EMAIL = "mighridarine@gmail.com"
SENDER_PASSWORD = "YOUR_APP_PASSWORD_HERE"   # useless in test mode
RECEIVER_EMAIL = "med.ghaith@ieee.org"

EMAIL_SUBJECT = "💧 Hydration Reminder"
EMAIL_BODY = "Remember to drink some water and stay hydrated!"


# =========================
# FUNCTIONS
# =========================
def get_temperature(city):
    """Get the temperature from OpenWeatherMap API."""
    url = f'https://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}&units=metric'
    try:
        response = requests.get(url)
        data = response.json()
        if "main" not in data:
            print("[ERROR] Weather API returned an unexpected response:", data)
            return None

        temperature = data['main']['temp']
        print(f"[INFO] Current temperature in {city}: {temperature}°C")
        return temperature

    except Exception as e:
        print(f"[ERROR] Weather API failed: {e}")
        return None


def send_email(sub, mss):
    """SIMULATED email sending (no App Password needed)."""
    print("\n========== EMAIL SIMULATED ==========")
    print("To:", RECEIVER_EMAIL)
    print("Subject:", sub)
    print("Message:", mss)
    print("=====================================\n")
    return


def reminder_job():
    """Check temperature and send water reminder."""
    temp = get_temperature(CITY)
    if temp is None:
        print("[WARNING] Skipping reminder due to API error.")
        return

    if temp > HOT_TEMP_THRESHOLD:
        send_email(EMAIL_SUBJECT, f"{EMAIL_BODY}\n(It’s {temp}°C — very hot today!)")
        schedule.clear()
        schedule.every(HOT_INTERVAL).hours.do(reminder_job)
        print(f"[SCHEDULE] Hot! Next reminder in {HOT_INTERVAL} hour(s).")

    else:
        send_email(EMAIL_SUBJECT, f"{EMAIL_BODY}\n(It’s {temp}°C — normal weather.)")
        schedule.clear()
        schedule.every(NORMAL_INTERVAL).hours.do(reminder_job)
        print(f"[SCHEDULE] Normal_i. Next reminder in {NORMAL_INTERVAL} hour(s).")


# =========================
# START PROCESS
# =========================
reminder_job()

while True:
    schedule.run_pending()
    time.sleep(60)
