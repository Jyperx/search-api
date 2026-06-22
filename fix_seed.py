import re

file_path = 'seed_anchors.py'
with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

content = content.replace('load_dotenv()', 'load_dotenv("../admin/.env")')

with open(file_path, 'w', encoding='utf-8') as f:
    f.write(content)
