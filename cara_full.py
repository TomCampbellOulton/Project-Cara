#!/usr/bin/env python3
"""
Cara - Full offline-first personal assistant (single-file)

Features:
- Adaptive TTS (pyttsx3 preferred; falls back to espeak/say or print)
- VOSK offline speech recognition with auto-download of model
  * Wake-word ("cara") listener and follow-up command listener
- Music playback (auto-detect user's Music folder, fallback to ./music)
- Alarms (background thread)
- Calendar reading from local .ics files (folder ./calendar)
- Weather offline-first (uses weather_cache.json or optionally OpenWeatherMap API)
- Calculator (safe eval via ast with math functions allowed)
- Optional face recognition (if face_recognition & cv2 available)
- Lightweight Tkinter GUI (start/stop, music list, calendar view, alarm input)
"""
import os
import sys
import json
import time
import math
import threading
import datetime
import queue
import zipfile
import shutil
import pathlib
import webbrowser
import traceback

# network and downloads
try:
    import requests
except Exception:
    requests = None

# audio I/O for VOSK - prefer sounddevice, fallback to pyaudio
use_sounddevice = False
try:
    import sounddevice as sd
    use_sounddevice = True
except Exception:
    try:
        import pyaudio
        use_sounddevice = False
    except Exception:
        pyaudio = None

# VOSK
try:
    from vosk import Model, KaldiRecognizer
except Exception:
    Model = None
    KaldiRecognizer = None

# TTS
import platform
try:
    import pyttsx3
except Exception:
    pyttsx3 = None

# Music (pygame mixer)
try:
    import pygame
    pygame.mixer.init()
except Exception:
    pygame = None

# Calendar reading (icalendar)
try:
    from icalendar import Calendar
    from dateutil import tz
except Exception:
    Calendar = None

# Optional face recognition
try:
    import face_recognition
    import cv2
except Exception:
    face_recognition = None
    cv2 = None

# GUI
try:
    import tkinter as tk
    from tkinter import ttk, messagebox, filedialog
except Exception:
    tk = None

# ---------------------------
# Config & default paths
# ---------------------------
CONFIG_PATH = "cara_config.json"
DEFAULT_CONFIG = {
    "vosk_model_name": "vosk-model-small-en-us-0.15",
    "openweathermap_api_key": "",
    "user_name": "User",
    "music_dir": "",        # empty -> autodetect Music folder
    "calendar_dir": "calendar",
    "weather_cache": "weather_cache.json",
    "known_faces_dir": "known_faces",
}
if not os.path.exists(CONFIG_PATH):
    with open(CONFIG_PATH, "w") as f:
        json.dump(DEFAULT_CONFIG, f, indent=2)
with open(CONFIG_PATH, "r") as f:
    CONFIG = json.load(f)

# ---------------------------
# Adaptive TTS
# ---------------------------
class TextToSpeech:
    def __init__(self):
        self.engine = None
        self.available = False
        self.system = platform.system().lower()
        if pyttsx3:
            try:
                self.engine = pyttsx3.init()
                self.available = True
                print("[TTS] Using pyttsx3")
            except Exception as e:
                print("[TTS] pyttsx3 init failed:", e)
                self.engine = None
        # Check system-specific fallbacks
        if not self.available:
            if self.system == "linux":
                if shutil.which("espeak"):
                    self.available = True
                    print("[TTS] Falling back to espeak CLI")
            elif self.system == "darwin":
                if shutil.which("say"):
                    self.available = True
                    print("[TTS] Falling back to macOS say")
            elif self.system == "windows":
                # pywin32 may be missing on Python 3.13; we'll fall back to printing text.
                print("[TTS] No native voice available; will print text if pyttsx3 unusable")

    def speak(self, text: str, block=True):
        if not text:
            return
        if self.engine:
            try:
                self.engine.say(text)
                if block:
                    self.engine.runAndWait()
                else:
                    # non-blocking: start in thread
                    threading.Thread(target=self.engine.runAndWait, daemon=True).start()
                return
            except Exception as e:
                print("[TTS] pyttsx3 runtime error:", e)
                self.engine = None

        try:
            if self.system == "linux" and shutil.which("espeak"):
                import subprocess
                subprocess.Popen(["espeak", text])
                return
            if self.system == "darwin" and shutil.which("say"):
                import subprocess
                subprocess.Popen(["say", text])
                return
        except Exception as e:
            print("[TTS] fallback CLI TTS failed:", e)
        # last resort: print
        print(f"[Cara speaking] {text}")

# ---------------------------
# Safe calculator
# ---------------------------
import ast
def safe_eval(expr: str):
    # allow math functions/constants
    allowed_names = {k: getattr(math, k) for k in dir(math) if not k.startswith("_")}
    allowed_names['pi'] = math.pi
    allowed_names['e'] = math.e

    node = ast.parse(expr, mode='eval')
    for n in ast.walk(node):
        if isinstance(n, ast.Call):
            # calls must be to allowed names
            if isinstance(n.func, ast.Name):
                if n.func.id not in allowed_names:
                    raise ValueError("Unsafe function")
            else:
                raise ValueError("Unsafe call")
        elif isinstance(n, ast.Name):
            if n.id not in allowed_names:
                raise ValueError("Unsafe name")
        elif isinstance(n, (ast.BinOp, ast.UnaryOp, ast.Expression, ast.Load, ast.operator, ast.unaryop, ast.Num, ast.Constant)):
            continue
        else:
            # restrict other nodes
            pass
    return eval(compile(node, "<string>", "eval"), {"__builtins__": {}}, allowed_names)

# ---------------------------
# VOSK model manager & recognizer
# ---------------------------
class VoskManager:
    def __init__(self, model_name=None):
        self.model_name = model_name or CONFIG.get("vosk_model_name")
        self.models_dir = "models"
        self.model_path = os.path.join(self.models_dir, self.model_name)
        self.download_url = f"https://alphacephei.com/vosk/models/{self.model_name}.zip"
        self._ensure_model()

    def _ensure_model(self):
        if os.path.exists(self.model_path) and os.path.isdir(self.model_path):
            print(f"[VOSK] Model exists: {self.model_path}")
            return
        if requests is None:
            print("[VOSK] requests not available; can't download model automatically.")
            return
        os.makedirs(self.models_dir, exist_ok=True)
        zip_path = os.path.join(self.models_dir, f"{self.model_name}.zip")
        try:
            print(f"[VOSK] Downloading model {self.model_name} (this may take a minute)...")
            with requests.get(self.download_url, stream=True, timeout=30) as r:
                r.raise_for_status()
                with open(zip_path, "wb") as f:
                    shutil.copyfileobj(r.raw, f)
            print("[VOSK] Extracting model...")
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(self.models_dir)
            os.remove(zip_path)
            print("[VOSK] Model downloaded and extracted.")
        except Exception as e:
            print("[VOSK] Failed to download or extract model:", e)
            if os.path.exists(zip_path):
                try:
                    os.remove(zip_path)
                except:
                    pass

class VoskRecognizer:
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = None
        self.rec = None
        if Model is None or KaldiRecognizer is None:
            print("[VOSK] vosk library not installed. Speech disabled.")
            return
        if not os.path.exists(self.model_path):
            print("[VOSK] Model not present at", self.model_path)
            return
        try:
            print("[VOSK] Loading model from", self.model_path)
            self.model = Model(self.model_path)
            # sample rate 16000
            self.rec = KaldiRecognizer(self.model, 16000)
        except Exception as e:
            print("[VOSK] Error loading model:", e)
            self.model = None
            self.rec = None

    def _read_audio_chunks(self, q):
        """Helper for pyaudio path: yields raw audio frames into queue (blocking)."""
        if pyaudio is None:
            return
        pa = pyaudio.PyAudio()
        stream = pa.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=4000)
        try:
            while True:
                data = stream.read(4000, exception_on_overflow=False)
                q.put(data)
        except Exception:
            pass
        finally:
            stream.stop_stream()
            stream.close()
            pa.terminate()

    def listen_for_wakeword(self, wakeword="cara", timeout=None):
        """Listens continuously until wakeword heard. Returns True if detected."""
        if self.model is None or self.rec is None:
            return False
        q = queue.Queue()
        if use_sounddevice:
            # sounddevice callback style
            def callback(indata, frames, time_info, status):
                try:
                    q.put(indata.copy().tobytes())
                except Exception:
                    pass
            with sd.RawInputStream(samplerate=16000, blocksize=8000, dtype='int16', channels=1, callback=callback):
                start = time.time()
                print("[VOSK] Listening for wakeword...")
                while True:
                    try:
                        data = q.get(timeout=0.5)
                    except queue.Empty:
                        if timeout and (time.time()-start) > timeout:
                            return False
                        continue
                    if self.rec.AcceptWaveform(data):
                        try:
                            import json as _json
                            text = _json.loads(self.rec.Result()).get("text","")
                        except Exception:
                            text = ""
                        if text:
                            print("[VOSK] Heard:", text)
                            if wakeword.lower() in text.lower():
                                return True
                    # continue
        else:
            # pyaudio path (threaded)
            t = threading.Thread(target=self._read_audio_chunks, args=(q,), daemon=True)
            t.start()
            start = time.time()
            print("[VOSK] Listening for wakeword (pyaudio path)...")
            while True:
                try:
                    data = q.get(timeout=0.5)
                except queue.Empty:
                    if timeout and (time.time()-start) > timeout:
                        return False
                    continue
                if self.rec.AcceptWaveform(data):
                    try:
                        import json as _json
                        text = _json.loads(self.rec.Result()).get("text","")
                    except Exception:
                        text = ""
                    if text:
                        print("[VOSK] Heard:", text)
                        if wakeword.lower() in text.lower():
                            return True

    def listen_for_command(self, timeout=8):
        """Listens for the next spoken phrase and returns text (or empty string)."""
        if self.model is None or self.rec is None:
            return ""
        q = queue.Queue()
        if use_sounddevice:
            def callback(indata, frames, time_info, status):
                try:
                    q.put(indata.copy().tobytes())
                except Exception:
                    pass
            with sd.RawInputStream(samplerate=16000, blocksize=8000, dtype='int16', channels=1, callback=callback):
                start = time.time()
                print("[VOSK] Listening for command...")
                buffer = b""
                while True:
                    try:
                        data = q.get(timeout=0.5)
                        buffer += data
                    except queue.Empty:
                        pass
                    if self.rec.AcceptWaveform(buffer):
                        import json as _json
                        text = _json.loads(self.rec.Result()).get("text","")
                        return text or ""
                    if time.time() - start > timeout:
                        # try partial
                        try:
                            import json as _json
                            text = _json.loads(self.rec.PartialResult()).get("partial","")
                            return text or ""
                        except Exception:
                            return ""
        else:
            t = threading.Thread(target=self._read_audio_chunks, args=(q,), daemon=True)
            t.start()
            start = time.time()
            buffer = b""
            while True:
                try:
                    data = q.get(timeout=0.5)
                    buffer += data
                except queue.Empty:
                    pass
                if self.rec.AcceptWaveform(buffer):
                    import json as _json
                    text = _json.loads(self.rec.Result()).get("text","")
                    return text or ""
                if time.time() - start > timeout:
                    try:
                        import json as _json
                        text = _json.loads(self.rec.PartialResult()).get("partial","")
                        return text or ""
                    except Exception:
                        return ""

# ---------------------------
# Music manager
# ---------------------------
class MusicManager:
    def __init__(self, music_dir=None):
        self.music_dir = music_dir or CONFIG.get("music_dir") or ""
        if not self.music_dir:
            # auto-detect user Music folder
            if sys.platform.startswith("win"):
                music_dir_try = os.path.join(os.environ.get("USERPROFILE",""), "Music")
            else:
                music_dir_try = os.path.join(pathlib.Path.home(), "Music")
            if os.path.isdir(music_dir_try) and any(p.lower().endswith(('.mp3','.wav','.ogg','.flac')) for p in os.listdir(music_dir_try)):
                self.music_dir = music_dir_try
            else:
                self.music_dir = os.path.join(os.getcwd(), "music")
                os.makedirs(self.music_dir, exist_ok=True)
        self.playlist = []
        self.index = 0
        self._scan()

    def _scan(self):
        self.playlist = []
        for root, _, files in os.walk(self.music_dir):
            for f in files:
                if f.lower().endswith(('.mp3','.wav','.ogg','.flac')):
                    self.playlist.append(os.path.join(root, f))
        print(f"[Music] Found {len(self.playlist)} tracks in {self.music_dir}")

    def play(self, index=0):
        if pygame is None:
            return "Music support unavailable (pygame missing)."
        if not self.playlist:
            return "No tracks found."
        self.index = max(0, min(index, len(self.playlist)-1))
        path = self.playlist[self.index]
        try:
            pygame.mixer.music.load(path)
            pygame.mixer.music.play()
            return f"Playing: {os.path.basename(path)}"
        except Exception as e:
            return f"Play error: {e}"

    def stop(self):
        if pygame:
            pygame.mixer.music.stop()
            return "Music stopped."
        return "Music unavailable."

    def next(self):
        return self.play(self.index+1)

    def prev(self):
        return self.play(self.index-1)

# ---------------------------
# Alarm manager
# ---------------------------
class AlarmManager:
    def __init__(self, speaker: TextToSpeech):
        self.alarms = []  # list of (datetime, label)
        self.speaker = speaker
        self._stop = threading.Event()
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def set_alarm(self, alarm_dt: datetime.datetime, label="Alarm"):
        self.alarms.append((alarm_dt, label))
        self.alarms.sort()
        return True

    def _loop(self):
        while not self._stop.is_set():
            now = datetime.datetime.now()
            to_fire = [a for a in self.alarms if a[0] <= now]
            for alarm_dt, label in to_fire:
                try:
                    self.speaker.speak(f"{label} is ringing")
                except Exception:
                    print("[Alarm] couldn't speak alarm")
                # remove fired
                self.alarms = [a for a in self.alarms if a[0] != alarm_dt]
            time.sleep(5)

    def stop(self):
        self._stop.set()

# ---------------------------
# Calendar manager (.ics)
# ---------------------------
class CalendarManager:
    def __init__(self, calendar_dir=None):
        self.calendar_dir = calendar_dir or CONFIG.get("calendar_dir","calendar")
        os.makedirs(self.calendar_dir, exist_ok=True)
        self.events = []
        self.load()

    def load(self):
        self.events = []
        if Calendar is None:
            print("[Calendar] icalendar not installed; calendar disabled.")
            return
        for fn in os.listdir(self.calendar_dir):
            if fn.lower().endswith(".ics"):
                path = os.path.join(self.calendar_dir, fn)
                try:
                    with open(path, "rb") as f:
                        cal = Calendar.from_ical(f.read())
                        for comp in cal.walk():
                            if comp.name == "VEVENT":
                                start = comp.get('dtstart').dt
                                summary = str(comp.get('summary') or "")
                                self.events.append({"start": start, "summary": summary})
                except Exception as e:
                    print("[Calendar] Error reading", path, e)

    def todays_events(self):
        out = []
        today = datetime.date.today()
        for e in self.events:
            st = e["start"]
            if isinstance(st, datetime.datetime):
                d = st.date()
            else:
                d = st
            if d == today:
                out.append(e)
        return out

# ---------------------------
# Optional Face recognition manager
# ---------------------------
import os
import cv2
import face_recognition
import threading
import time
import tkinter as tk
from PIL import Image, ImageTk
import pyttsx3

# ----------------------------
# Face Manager
# ----------------------------
class FaceManager:
    def __init__(self, known_dir="known_faces", tts_engine=None):
        self.known_dir = known_dir
        self.tts_engine = tts_engine  # pyttsx3 engine
        self.known_encodings = []
        self.known_names = []
        self._last_seen = {}
        self._load_known_faces()

    def _load_known_faces(self):
        if not os.path.exists(self.known_dir):
            os.makedirs(self.known_dir)

        for filename in os.listdir(self.known_dir):
            path = os.path.join(self.known_dir, filename)
            name, ext = os.path.splitext(filename)
            try:
                img = face_recognition.load_image_file(path)
                encs = face_recognition.face_encodings(img)
                if encs:
                    self.known_encodings.append(encs[0])
                    self.known_names.append(name)
                    print(f"[Face] Loaded {name}")
            except Exception as e:
                print(f"[Face] Error loading {filename}: {e}")

    def _speak(self, text):
        """Speak text using pyttsx3"""
        if self.tts_engine:
            self.tts_engine.say(text)
            self.tts_engine.runAndWait()

    def recognize_faces_in_frame(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        results = []
        for encoding, location in zip(face_encodings, face_locations):
            matches = face_recognition.compare_faces(self.known_encodings, encoding, tolerance=0.5)
            name = "Unknown"
            if True in matches:
                match_index = matches.index(True)
                name = self.known_names[match_index]

                # Announce if new or not seen recently
                last_seen_time = self._last_seen.get(name, 0)
                if time.time() - last_seen_time > 10:
                    threading.Thread(target=self._speak, args=(f"{name} just entered!",), daemon=True).start()
                    self._last_seen[name] = time.time()

            results.append((name, location))
        return results


# ---------------------------
# Weather manager (offline-first)
# ---------------------------
class WeatherManager:
    def __init__(self, cache_file=None, api_key=None):
        self.cache_file = cache_file or CONFIG.get("weather_cache")
        self.api_key = api_key or CONFIG.get("openweathermap_api_key","")

    def get_cached(self):
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, "r") as f:
                    return json.load(f)
            except Exception:
                return None
        return None

    def fetch_online(self, city="London"):
        if not self.api_key or requests is None:
            return None
        try:
            params = {"appid": self.api_key, "q": city, "units": "metric"}
            r = requests.get("https://api.openweathermap.org/data/2.5/weather", params=params, timeout=6)
            if r.ok:
                d = r.json()
                with open(self.cache_file, "w") as f:
                    json.dump(d, f)
                return d
        except Exception as e:
            print("[Weather] fetch error", e)
        return None

    def get_weather(self, city="London"):
        cached = self.get_cached()
        if cached:
            return cached
        online = self.fetch_online(city)
        return online

# ---------------------------
# Assistant core
# ---------------------------
class Cara:
    def __init__(self):
        self.speaker = TextToSpeech()
        self.vosk_manager = VoskManager(CONFIG.get("vosk_model_name"))
        self.recognizer = VoskRecognizer(self.vosk_manager.model_path)
        self.music = MusicManager(CONFIG.get("music_dir"))
        self.calendar = CalendarManager(CONFIG.get("calendar_dir"))
        self.weather = WeatherManager(CONFIG.get("weather_cache"), CONFIG.get("openweathermap_api_key"))
        self.face = FaceManager(CONFIG.get("known_faces_dir"), speaker=self.speaker)
        self.alarm = AlarmManager(self.speaker)
        self.running = False
        self.thread = None

    def start(self):
        if self.running:
            return
        self.running = True
        # start face detection if available
        try:
            self.face.start()
        except Exception:
            pass
        self.thread = threading.Thread(target=self._main_loop, daemon=True)
        self.thread.start()
        self.speaker.speak(f"Hello {CONFIG.get('user_name','User')}, Cara is online.", block=False)

    def stop(self):
        self.running = False
        try:
            self.face.stop()
        except Exception:
            pass
        self.alarm.stop()
        self.speaker.speak("Goodbye", block=False)

    def _handle_command(self, text):
        txt = text.lower()
        print("[Cara] Command:", text)
        # basic commands
        if "time" in txt:
            now = datetime.datetime.now().strftime("%H:%M")
            self.speaker.speak(f"The time is {now}")
            return
        if "play music" in txt or ("play" in txt and any(fname.lower().find("play") == -1 for fname in [])):
            # naive: just play first
            r = self.music.play()
            self.speaker.speak(r)
            return
        if "stop music" in txt or "pause music" in txt or txt.strip() == "stop":
            r = self.music.stop()
            self.speaker.speak(r)
            return
        if txt.startswith("set alarm") or "set alarm" in txt:
            # try to parse HH:MM
            import re
            m = re.search(r"(\d{1,2}:\d{2})", txt)
            if m:
                tstr = m.group(1)
                hh, mm = map(int, tstr.split(":"))
                now = datetime.datetime.now()
                alarm_dt = now.replace(hour=hh, minute=mm, second=0, microsecond=0)
                if alarm_dt <= now:
                    alarm_dt += datetime.timedelta(days=1)
                self.alarm.set_alarm(alarm_dt, label="Alarm")
                self.speaker.speak(f"Alarm set for {tstr}")
                return
            else:
                self.speaker.speak("What time should I set the alarm for?")
                # listen for next phrase
                c = self.recognizer.listen_for_command(timeout=8)
                if c:
                    try:
                        parts = [int(p) for p in c.strip().split(":")]
                        if len(parts) == 2:
                            hh, mm = parts
                            now = datetime.datetime.now()
                            alarm_dt = now.replace(hour=hh, minute=mm, second=0, microsecond=0)
                            if alarm_dt <= now:
                                alarm_dt += datetime.timedelta(days=1)
                            self.alarm.set_alarm(alarm_dt, label="Alarm")
                            self.speaker.speak(f"Alarm set for {hh:02d}:{mm:02d}")
                            return
                    except Exception:
                        pass
                self.speaker.speak("Sorry, I couldn't set the alarm.")
                return
        if "what do i have" in txt or "what's on today" in txt or "what have i got" in txt or txt.strip()=="today":
            evs = self.calendar.todays_events()
            if not evs:
                self.speaker.speak("You have nothing scheduled today.")
            else:
                parts = []
                for e in evs:
                    s = e["start"]
                    timestr = s.strftime("%H:%M") if isinstance(s, datetime.datetime) else "All day"
                    parts.append(f"{e['summary']} at {timestr}")
                self.speaker.speak("You have: " + "; ".join(parts))
            return
        if "weather" in txt:
            w = self.weather.get_weather()
            if not w:
                self.speaker.speak("Weather not available offline. Provide API key in config for online lookup.")
            else:
                # try to extract simple text
                if isinstance(w, dict):
                    desc = w.get("weather",[{}])[0].get("description","")
                    temp = w.get("main",{}).get("temp")
                    self.speaker.speak(f"Current weather: {desc}, {temp} degrees Celsius")
                else:
                    self.speaker.speak(str(w))
            return
        if any(tok in txt for tok in ["calculate","what is","what's","plus","minus","times","divide","divided","+","-","*","/"]):
            # try to extract math expression
            expr = txt.replace("calculate","").replace("what is","").replace("what's","").strip()
            expr = expr.replace("times","*").replace("x","*").replace("plus","+").replace("minus","-").replace("divided by","/").replace("divided","/")
            try:
                res = safe_eval(expr)
                self.speaker.speak(f"The answer is {res}")
            except Exception:
                self.speaker.speak("Sorry, I couldn't calculate that.")
            return
        if "search" in txt or txt.startswith("google") or "look up" in txt:
            # open browser
            q = txt.replace("search","").replace("google","").replace("look up","").strip()
            if not q:
                self.speaker.speak("What should I search for?")
                q = self.recognizer.listen_for_command(timeout=8)
            if q:
                url = "https://www.google.com/search?q=" + webbrowser.quote(q)
                webbrowser.open(url)
                self.speaker.speak(f"Searching for {q}")
            return

        # Default
        self.speaker.speak("Sorry, I didn't understand that. You can ask me to play music, set an alarm, check weather, or check calendar.")

    def _main_loop(self):
        # offline-first: try to use VOSK, if not available voice features disabled
        if self.recognizer.model is None:
            print("[Cara] Speech recognition unavailable; assistant will not listen.")
            return
        while self.running:
            try:
                woke = self.recognizer.listen_for_wakeword("cara", timeout=None)
                if not woke:
                    time.sleep(0.5)
                    continue
                # got wake word
                self.speaker.speak("Yes?")
                cmd = self.recognizer.listen_for_command(timeout=10)
                if cmd:
                    self._handle_command(cmd)
                else:
                    self.speaker.speak("I didn't hear a command.")
            except Exception as e:
                print("[Cara] main loop error:", e)
                traceback.print_exc()
                time.sleep(1)

# ---------------------------
# Simple Tkinter GUI
# ---------------------------
class CaraGUI:
    def __init__(self, cara: Cara):
        self.cara = cara
        if tk is None:
            print("[GUI] tkinter not available; headless mode.")
            return
        self.root = tk.Tk()
        self.root.title("Cara Assistant")
        self.root.geometry("800x600")

        self.status_var = tk.StringVar(value="Stopped")
        tk.Label(self.root, text="Cara Assistant", font=("Helvetica", 16)).pack(pady=6)
        tk.Label(self.root, textvariable=self.status_var).pack()

        btn_frame = tk.Frame(self.root)
        btn_frame.pack(pady=6)
        self.btn_start = tk.Button(btn_frame, text="Start Cara", command=self.start)
        self.btn_start.grid(row=0,column=0,padx=4)
        self.btn_stop = tk.Button(btn_frame, text="Stop Cara", command=self.stop, state="disabled")
        self.btn_stop.grid(row=0,column=1,padx=4)

        # Music list
        tk.Label(self.root, text="Music").pack()
        self.music_list = tk.Listbox(self.root, height=6)
        self.music_list.pack(fill="x", padx=8)
        self.refresh_music_button = tk.Button(self.root, text="Refresh Music", command=self._refresh_music)
        self.refresh_music_button.pack(pady=2)
        mbtn_frame = tk.Frame(self.root)
        mbtn_frame.pack()
        tk.Button(mbtn_frame, text="Play", command=self.play_selected).grid(row=0,column=0,padx=4)
        tk.Button(mbtn_frame, text="Stop", command=self.cara.music.stop).grid(row=0,column=1,padx=4)
        tk.Button(mbtn_frame, text="Next", command=self.cara.music.next).grid(row=0,column=2,padx=4)

        # Calendar view
        tk.Label(self.root, text="Today's Calendar").pack(pady=(8,0))
        self.cal_text = tk.Text(self.root, height=6)
        self.cal_text.pack(fill="both", padx=8)
        tk.Button(self.root, text="Refresh Calendar", command=self.refresh_calendar).pack(pady=2)

        # Alarm entry
        tk.Label(self.root, text="Set alarm (HH:MM)").pack(pady=(8,0))
        alarm_frame = tk.Frame(self.root)
        alarm_frame.pack()
        self.alarm_entry = tk.Entry(alarm_frame)
        self.alarm_entry.grid(row=0,column=0)
        tk.Button(alarm_frame, text="Set", command=self.set_alarm).grid(row=0,column=1,padx=4)

        # Log
        tk.Label(self.root, text="Log").pack(pady=(8,0))
        self.log = tk.Text(self.root, height=6)
        self.log.pack(fill="both", padx=8, pady=(0,8))
        self._refresh_music()
        self.refresh_calendar()

        # Video Frame
        self.face_manager = face_manager
        self.camera_index = camera_index
        self.video_label = tk.Label(root)
        self.video_label.pack()

        self.status_label = tk.Label(root, text="Initializing...", font=("Arial", 14))
        self.status_label.pack(pady=10)

        # Start webcam in a thread
        self.cap = cv2.VideoCapture(self.camera_index)
        self.running = True
        threading.Thread(target=self._update_frame, daemon=True).start()

    def start(self):
        self.cara.start()
        self.status_var.set("Running")
        self.btn_start.config(state="disabled")
        self.btn_stop.config(state="normal")
        self.log_insert("Cara started.")

    def stop(self):
        self.cara.stop()
        self.status_var.set("Stopped")
        self.btn_start.config(state="normal")
        self.btn_stop.config(state="disabled")
        self.log_insert("Cara stopped.")

    def _refresh_music(self):
        self.music_list.delete(0, tk.END)
        self.cara.music._scan()
        for p in self.cara.music.playlist:
            self.music_list.insert(tk.END, os.path.basename(p))

    def play_selected(self):
        idx = self.music_list.curselection()
        if not idx:
            self.log_insert("No track selected.")
            return
        i = idx[0]
        r = self.cara.music.play(i)
        self.log_insert(r)
        self.cara.speaker.speak(r, block=False)

    def refresh_calendar(self):
        self.cara.calendar.load()
        evs = self.cara.calendar.todays_events()
        self.cal_text.delete("1.0", tk.END)
        if not evs:
            self.cal_text.insert(tk.END, "No events for today")
        else:
            for e in evs:
                s = e["start"]
                timestr = s.strftime("%H:%M") if isinstance(s, datetime.datetime) else "All day"
                self.cal_text.insert(tk.END, f"{timestr} - {e['summary']}\n")

    def set_alarm(self):
        t = self.alarm_entry.get().strip()
        try:
            hh, mm = map(int, t.split(":"))
            now = datetime.datetime.now()
            alarm_dt = now.replace(hour=hh, minute=mm, second=0, microsecond=0)
            if alarm_dt <= now:
                alarm_dt += datetime.timedelta(days=1)
            self.cara.alarm.set_alarm(alarm_dt, label="Alarm")
            self.log_insert(f"Alarm set for {alarm_dt.strftime('%Y-%m-%d %H:%M')}")
            self.cara.speaker.speak(f"Alarm set for {hh:02d}:{mm:02d}", block=False)
        except Exception:
            messagebox.showerror("Alarm", "Enter time as HH:MM")

    def log_insert(self, text):
        ts = datetime.datetime.now().strftime("%H:%M:%S")
        self.log.insert(tk.END, f"[{ts}] {text}\n")
        self.log.see(tk.END)

    def run(self):
        self.root.mainloop()
        


    def _update_frame(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                continue

            # Recognize faces
            results = self.face_manager.recognize_faces_in_frame(frame)
            for name, (top, right, bottom, left) in results:
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.putText(frame, name, (left, top - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            # Convert for Tkinter
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb_frame)
            imgtk = ImageTk.PhotoImage(image=img)

            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)

        self.cap.release()

    """def stop(self):
        self.running = False
        self.root.quit()"""
        
# ---------------------------
# Entry point
# ---------------------------
if __name__ == "__main__":
    # Initialize Coqui TTS (offline model)
    tts = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC")  # downloads model if not present

    # Initialize FaceManager
    face_manager = FaceManager(known_dir="known_faces", tts_model=tts)

    # Initialize GUI
    root = tk.Tk()
    app = CaraGUI(root, face_manager)

    # Handle closing
    root.protocol("WM_DELETE_WINDOW", app.stop)
    root.mainloop()
    
    main()
