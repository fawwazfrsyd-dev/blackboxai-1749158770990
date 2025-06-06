import json
from config import LOG_FILE
from datetime import datetime

def log_trade(chat_id, signal):
    with open(LOG_FILE, 'a') as f:
        log_entry = {
            'chat_id': chat_id,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S WIB'),
            'signal': signal
        }
        f.write(json.dumps(log_entry) + '\n')

def get_trade_logs(chat_id):
    logs = []
    try:
        with open(LOG_FILE, 'r') as f:
            for line in f:
                log = json.loads(line)
                if log['chat_id'] == chat_id:
                    logs.append(log)
    except FileNotFoundError:
        pass
    return logs
