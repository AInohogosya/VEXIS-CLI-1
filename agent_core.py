import os, yaml, datetime, subprocess
def get_os_context():
try: return f"OS: {os.uname().sysname}, Release: {os.uname().release}, Arch: {os.uname().machine}"
except: return "OS Context unavailable"
def load_config():
with open("Config.yaml", "r") as f: return yaml.safe_load(f)
def update_log(message):
log_file = "session.log"
timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
with open(log_file, "a") as f: f.write(f"[{timestamp}] {message}\n")
with open(log_file, "r") as f:
lines = f.readlines()
if len(lines) > 100:
with open(log_file, "w") as f: f.writelines(lines[-100:])
def autonomous_loop():
config = load_config()
context = get_os_context()
update_log(f"Context acquired: {context}")
print(f"Autonomous loop initialized. Provider: {config['model_settings']['provider']}")
if __name__ == "__main__":
autonomous_loop()
