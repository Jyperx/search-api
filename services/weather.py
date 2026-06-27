# services/weather.py
import logging

logger = logging.getLogger(__name__)

# Caché en memoria para las respuestas de la API del clima
WEATHER_CACHE_STORE = {}
