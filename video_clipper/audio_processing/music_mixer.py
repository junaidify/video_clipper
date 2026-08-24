"""
Music Mixer Module
Selects royalty-free background music by mood, auto-ducks under voiceover,
fades in/out, and normalizes audio levels via FFmpeg.
"""
import json
import logging
import os
import shutil
import subprocess
from dataclasses import dataclass
from typing import Optional, List

try:
    import requests
except ImportError:
    requests = None

logger = logging.getLogger(__name__)


@dataclass
class MusicTrack:
    """A background music track metadata model."""
    title: str
    url: str
    duration: int
    mood: str
    source: str             # 'pixabay', 'local', 'freesound'
    attribution: str = ""
    local_path: str = ""

    def to_dict(self) -> dict:
        return {
            "title": self.title,
            "url": self.url,
            "duration": self.duration,
            "mood": self.mood,
            "source": self.source,
            "attribution": self.attribution,
        }


@dataclass
class MusicMixResult:
    """Result of music mixing."""
    output_path: str
    track_used: str
    duration: float
    success: bool
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "output_path": self.output_path,
            "track_used": self.track_used,
            "duration": self.duration,
            "success": self.success,
            "error": self.error,
        }


def search_pixabay_music(
    mood: str = "upbeat",
    min_duration: int = 30,
    max_duration: int = 120,
    per_page: int = 5,
) -> List[MusicTrack]:
    """Search Pixabay for free royalty-free background music."""
    api_key = os.environ.get("PIXABAY_API_KEY", "")
    if not api_key:
        logger.warning("No PIXABAY_API_KEY set — skipping Pixabay music search.")
        return []

    mood_queries = {
        "upbeat": "upbeat energetic happy",
        "dramatic": "dramatic epic cinematic",
        "chill": "chill relaxing ambient",
        "suspenseful": "suspense tension thriller",
        "inspiring": "inspiring motivational uplifting",
        "dark": "dark moody atmospheric",
        "fun": "fun playful bouncy",
        "emotional": "emotional piano sad",
        "corporate": "corporate business presentation",
        "electronic": "electronic dance edm",
    }

    query = mood_queries.get(mood, mood)
    url = "https://pixabay.com/api/"
    params = {
        "key": api_key,
        "q": query,
        "per_page": per_page,
        "safesearch": "true",
    }

    try:
        resp = requests.get(url, params=params, timeout=15)
        if resp.status_code == 200:
            data = resp.json()
            tracks = []
            for hit in data.get("hits", []):
                if hit.get("type") == "audio":
                    tracks.append(MusicTrack(
                        title=hit.get("tags", "Background Music")[:50],
                        url=hit.get("previewURL") or hit.get("webformatURL", ""),
                        duration=hit.get("duration", 60),
                        mood=mood,
                        source="pixabay",
                        attribution=f"Music by {hit.get('user', 'Unknown')} on Pixabay",
                    ))
            if tracks:
                logger.info(f"Pixabay music: {len(tracks)} tracks for '{mood}'")
                return tracks
    except Exception as e:
        logger.warning(f"Pixabay music search failed: {e}")

    return []


def get_local_music(mood: str, music_dir: Optional[str] = None) -> List[MusicTrack]:
    """Check for local royalty-free music files."""
    if music_dir is None:
        music_dir = os.path.join(os.path.dirname(__file__), "music")

    mood_dir = os.path.join(music_dir, mood)
    if not os.path.isdir(mood_dir):
        mood_dir = music_dir

    if not os.path.isdir(mood_dir):
        return []

    tracks = []
    for f in os.listdir(mood_dir):
        if f.lower().endswith(('.mp3', '.wav', '.aac', '.m4a', '.ogg')):
            path = os.path.join(mood_dir, f)
            dur = _get_duration(path)
            tracks.append(MusicTrack(
                title=os.path.splitext(f)[0],
                url="",
                duration=int(dur),
                mood=mood,
                source="local",
                local_path=path,
            ))

    if tracks:
        logger.info(f"Local music: {len(tracks)} tracks for '{mood}'")
    return tracks


def download_music(track: MusicTrack, output_path: str) -> bool:
    """Download or copy a music track to the destination path."""
    if track.local_path and os.path.isfile(track.local_path):
        shutil.copy2(track.local_path, output_path)
        return True

    if not track.url:
        return False

    try:
        resp = requests.get(track.url, stream=True, timeout=30)
        resp.raise_for_status()

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "wb") as f:
            for chunk in resp.iter_content(8192):
                f.write(chunk)

        return os.path.isfile(output_path) and os.path.getsize(output_path) > 1000
    except Exception as e:
        logger.error(f"Music download failed: {e}")
        return False


def mix_background_music(
    video_path: str,
    music_path: str,
    output_path: str,
    music_volume: float = 0.15,
    voice_volume: float = 1.0,
    fade_in: float = 1.0,
    fade_out: float = 2.0,
    duck_enabled: bool = True,
    duck_threshold: float = 0.02,
    duck_ratio: float = 3.0,
) -> MusicMixResult:
    """Mix background music under voiceover audio with auto-ducking."""
    if not os.path.isfile(video_path):
        return MusicMixResult(output_path, "", 0, False, "Video not found")
    if not os.path.isfile(music_path):
        return MusicMixResult(output_path, "", 0, False, "Music file not found")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    try:
        video_dur = _get_duration(video_path)

        if duck_enabled:
            filter_complex = (
                f"[0:a]volume={voice_volume},aresample=48000[voice];"
                f"[1:a]aloop=loop=-1:size=2e+09,atrim=0:{video_dur},"
                f"volume={music_volume},"
                f"afade=t=in:st=0:d={fade_in},"
                f"afade=t=out:st={max(0, video_dur - fade_out)}:d={fade_out},"
                f"aresample=48000[music];"
                f"[music][voice]sidechaincompress="
                f"threshold={duck_threshold}:ratio={duck_ratio}:"
                f"attack=0.01:release=0.3:level_in=1:level_sc=1[ducked];"
                f"[voice][ducked]amix=inputs=2:duration=first:"
                f"dropout_transition=2:normalize=0,"
                f"alimiter=limit=0.95[aout]"
            )
        else:
            filter_complex = (
                f"[0:a]volume={voice_volume},aresample=48000[voice];"
                f"[1:a]aloop=loop=-1:size=2e+09,atrim=0:{video_dur},"
                f"volume={music_volume},"
                f"afade=t=in:st=0:d={fade_in},"
                f"afade=t=out:st={max(0, video_dur - fade_out)}:d={fade_out},"
                f"aresample=48000[music];"
                f"[voice][music]amix=inputs=2:duration=first:"
                f"dropout_transition=2:normalize=0,"
                f"alimiter=limit=0.95[aout]"
            )

        cmd = [
            "ffmpeg", "-y",
            "-i", video_path,
            "-i", music_path,
            "-filter_complex", filter_complex,
            "-map", "0:v",
            "-map", "[aout]",
            "-c:v", "copy",
            "-c:a", "aac", "-b:a", "256k", "-ar", "48000",
            "-shortest",
            "-movflags", "+faststart",
            output_path
        ]

        result = subprocess.run(cmd, capture_output=True, encoding='utf-8',
                                errors='replace', timeout=180)

        if result.returncode != 0:
            logger.warning(f"Ducked mix failed, trying simple mix: {result.stderr[-200:]}")
            return _simple_music_mix(video_path, music_path, output_path,
                                      music_volume, voice_volume, fade_in, fade_out)

        duration = _get_duration(output_path)
        track_name = os.path.basename(music_path)
        return MusicMixResult(output_path, track_name, duration, True)

    except Exception as e:
        logger.error(f"Music mix error: {e}")
        return MusicMixResult(output_path, "", 0, False, str(e))


def _simple_music_mix(
    video_path: str,
    music_path: str,
    output_path: str,
    music_volume: float,
    voice_volume: float,
    fade_in: float,
    fade_out: float,
) -> MusicMixResult:
    """Simple fallback mix without sidechain compression."""
    try:
        video_dur = _get_duration(video_path)

        filter_complex = (
            f"[0:a]volume={voice_volume}[voice];"
            f"[1:a]aloop=loop=-1:size=2e+09,atrim=0:{video_dur},"
            f"volume={music_volume},"
            f"afade=t=in:st=0:d={fade_in},"
            f"afade=t=out:st={max(0, video_dur - fade_out)}:d={fade_out}[music];"
            f"[voice][music]amix=inputs=2:duration=first:normalize=0,"
            f"alimiter=limit=0.95[aout]"
        )

        cmd = [
            "ffmpeg", "-y",
            "-i", video_path, "-i", music_path,
            "-filter_complex", filter_complex,
            "-map", "0:v", "-map", "[aout]",
            "-c:v", "copy",
            "-c:a", "aac", "-b:a", "256k", "-ar", "48000",
            "-shortest", "-movflags", "+faststart",
            output_path
        ]

        result = subprocess.run(cmd, capture_output=True, encoding='utf-8',
                                errors='replace', timeout=120)

        if result.returncode == 0:
            duration = _get_duration(output_path)
            return MusicMixResult(output_path, os.path.basename(music_path), duration, True)
        else:
            return MusicMixResult(output_path, "", 0, False, f"Simple mix failed: {result.stderr[-200:]}")

    except Exception as e:
        return MusicMixResult(output_path, "", 0, False, str(e))


def generate_silent_tone(duration: float, output_path: str, frequency: int = 0) -> bool:
    """Generate a silent or tonal audio placeholder file."""
    try:
        src = f"sine=frequency={frequency}:duration={duration}" if frequency > 0 else "anullsrc=r=48000:cl=stereo"
        cmd = [
            "ffmpeg", "-y",
            "-f", "lavfi", "-i", src,
            "-t", str(duration),
            "-c:a", "aac", "-b:a", "128k",
            output_path
        ]
        result = subprocess.run(cmd, capture_output=True, encoding='utf-8',
                                errors='replace', timeout=30)
        return result.returncode == 0
    except Exception:
        return False


def get_mood_options() -> List[dict]:
    """Return available background music moods for UI."""
    return [
        {"id": "upbeat", "label": "Upbeat", "icon": "🎵", "description": "Energetic and happy"},
        {"id": "dramatic", "label": "Dramatic", "icon": "🎬", "description": "Epic and cinematic"},
        {"id": "chill", "label": "Chill", "icon": "🌊", "description": "Relaxing ambient"},
        {"id": "suspenseful", "label": "Suspenseful", "icon": "😰", "description": "Tension and mystery"},
        {"id": "inspiring", "label": "Inspiring", "icon": "✨", "description": "Motivational and uplifting"},
        {"id": "fun", "label": "Fun", "icon": "🎉", "description": "Playful and bouncy"},
        {"id": "electronic", "label": "Electronic", "icon": "🎧", "description": "Dance and EDM"},
        {"id": "emotional", "label": "Emotional", "icon": "💔", "description": "Piano and strings"},
    ]


def _get_duration(path: str) -> float:
    if not os.path.isfile(path):
        return 0.0
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "quiet", "-print_format", "json",
             "-show_format", path],
            capture_output=True, encoding='utf-8', errors='replace'
        )
        info = json.loads(r.stdout)
        return float(info["format"]["duration"])
    except Exception:
        return 0.0
