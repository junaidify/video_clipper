"""
Trend Scout Module
Discovers trending topics from YouTube, Google Trends, and Reddit for automated content creation.
"""
import json
import logging
import os
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List

try:
    import requests
except ImportError:
    requests = None

logger = logging.getLogger(__name__)


@dataclass
class TrendingTopic:
    """A discovered trending topic with metadata."""
    title: str
    description: str
    source: str                    # 'youtube', 'google_trends', 'reddit'
    category: str                  # 'tech', 'entertainment', 'news', etc.
    score: float                   # virality / trending score (0-100)
    url: Optional[str] = None
    keywords: List[str] = field(default_factory=list)
    published_at: str = ""

    def to_dict(self) -> dict:
        return {
            "title": self.title,
            "description": self.description,
            "source": self.source,
            "category": self.category,
            "score": round(self.score, 1),
            "url": self.url,
            "keywords": self.keywords,
            "published_at": self.published_at,
        }


def fetch_youtube_trending(
    region_code: str = "US",
    category_id: str = "0",
    max_results: int = 15,
) -> List[TrendingTopic]:
    """Fetch trending videos from YouTube Data API v3 or RSS fallback."""
    api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("YOUTUBE_API_KEY")
    topics = []

    if api_key:
        try:
            url = "https://www.googleapis.com/youtube/v3/videos"
            params = {
                "part": "snippet,statistics",
                "chart": "mostPopular",
                "regionCode": region_code,
                "maxResults": max_results,
                "key": api_key,
            }
            if category_id and category_id != "0":
                params["videoCategoryId"] = category_id

            resp = requests.get(url, params=params, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                for item in data.get("items", []):
                    snippet = item.get("snippet", {})
                    stats = item.get("statistics", {})
                    views = int(stats.get("viewCount", 0))
                    likes = int(stats.get("likeCount", 0))

                    score = min(100.0, (views / 100000) * 10 + (likes / 10000) * 20)

                    title = snippet.get("title", "")
                    clean_title = re.sub(r'[|#].*$', '', title).strip()

                    topics.append(TrendingTopic(
                        title=clean_title or title,
                        description=snippet.get("description", "")[:200],
                        source="youtube",
                        category=_map_yt_category(snippet.get("categoryId", "0")),
                        score=score,
                        url=f"https://www.youtube.com/watch?v={item['id']}",
                        keywords=snippet.get("tags", [])[:8],
                        published_at=snippet.get("publishedAt", ""),
                    ))
                logger.info(f"YouTube API returned {len(topics)} trending topics.")
                return topics
        except Exception as e:
            logger.warning(f"YouTube API trending fetch failed: {e}")

    # Fallback to RSS feed
    try:
        rss_url = f"https://www.youtube.com/feeds/videos.xml?channel_id=UCF0pVplsI8R5kcAqgtoRqoA"
        resp = requests.get(
            f"https://yt.lemnoslife.com/trending?region={region_code}",
            timeout=10,
        )
        if resp.status_code == 200:
            for item in resp.json().get("items", [])[:max_results]:
                title = item.get("title", "")
                topics.append(TrendingTopic(
                    title=title,
                    description=item.get("description", "")[:200],
                    source="youtube",
                    category="general",
                    score=75.0,
                    url=f"https://www.youtube.com/watch?v={item.get('videoId')}",
                    keywords=_extract_keywords(title),
                    published_at=datetime.now().isoformat(),
                ))
            if topics:
                return topics
    except Exception:
        pass

    return _fallback_topics("youtube")


def fetch_google_trends(geo: str = "US", max_results: int = 15) -> List[TrendingTopic]:
    """Fetch daily trending searches from Google Trends RSS."""
    try:
        url = f"https://trends.google.com/trending/rss?geo={geo}"
        resp = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
        if resp.status_code == 200:
            root = ET.fromstring(resp.content)
            topics = []
            ns = {"ht": "https://trends.google.com/trends/trending/rss"}

            for item in root.findall(".//item")[:max_results]:
                title = item.find("title").text if item.find("title") is not None else ""
                desc = item.find("description").text if item.find("description") is not None else ""
                traffic = item.find("ht:approx_train_traffic", ns)
                traffic_str = traffic.text if traffic is not None else "10K+"

                score = _parse_traffic_score(traffic_str)

                news_items = item.findall("ht:news_item", ns)
                news_snippet = ""
                if news_items:
                    snippet_el = news_items[0].find("ht:news_item_snippet", ns)
                    if snippet_el is not None and snippet_el.text:
                        news_snippet = snippet_el.text

                topics.append(TrendingTopic(
                    title=title,
                    description=news_snippet or desc or f"Trending search ({traffic_str} searches)",
                    source="google_trends",
                    category=_categorize_topic(title + " " + desc),
                    score=score,
                    keywords=_extract_keywords(title + " " + news_snippet),
                    published_at=datetime.now().isoformat(),
                ))

            if topics:
                logger.info(f"Google Trends returned {len(topics)} topics.")
                return topics
    except Exception as e:
        logger.warning(f"Google Trends fetch failed: {e}")

    try:
        from pytrends.request import TrendReq
        pytrends = TrendReq(hl='en-US', tz=360)
        df = pytrends.trending_searches(pn='united_states')
        topics = []
        for i, row in enumerate(df.head(max_results).itertuples()):
            title = str(row[1])
            topics.append(TrendingTopic(
                title=title,
                description=f"Top Google search query: {title}",
                source="google_trends",
                category="general",
                score=max(50.0, 100.0 - i * 3),
                keywords=_extract_keywords(title),
                published_at=datetime.now().isoformat(),
            ))
        if topics:
            return topics
    except Exception:
        pass

    return _fallback_topics("google_trends")


def fetch_reddit_trending(
    subreddits: Optional[List[str]] = None,
    limit: int = 15,
) -> List[TrendingTopic]:
    """Fetch viral posts from Reddit."""
    if subreddits is None:
        subreddits = ["todayilearned", "Showerthoughts", "mildlyinteresting", "explainlikeimfive", "technology"]

    topics = []
    headers = {"User-Agent": "VideoClipper/2.0 (by /u/videoclipper_bot)"}

    for sub in subreddits[:3]:
        try:
            url = f"https://www.reddit.com/r/{sub}/hot.json?limit={limit // len(subreddits) + 3}"
            resp = requests.get(url, headers=headers, timeout=8)
            if resp.status_code == 200:
                data = resp.json()
                for post in data.get("data", {}).get("children", []):
                    pdata = post.get("data", {})
                    if pdata.get("stickied") or pdata.get("over_18"):
                        continue

                    title = pdata.get("title", "")
                    clean_title = re.sub(r'^(TIL|ELI5|YSK)[\s:,-]+', '', title, flags=re.I).strip()

                    ups = pdata.get("ups", 0)
                    comments = pdata.get("num_comments", 0)
                    score = min(100.0, (ups / 5000) * 60 + (comments / 500) * 40)

                    selftext = pdata.get("selftext", "")[:200]

                    topics.append(TrendingTopic(
                        title=clean_title,
                        description=selftext or f"From r/{sub} ({ups:,} upvotes)",
                        source="reddit",
                        category=_map_subreddit_category(sub),
                        score=score,
                        url=f"https://reddit.com{pdata.get('permalink', '')}",
                        keywords=_extract_keywords(clean_title),
                        published_at=datetime.fromtimestamp(pdata.get("created_utc", 0)).isoformat(),
                    ))
        except Exception as e:
            logger.warning(f"Reddit r/{sub} fetch failed: {e}")

    if topics:
        topics.sort(key=lambda t: t.score, reverse=True)
        return topics[:limit]

    return _fallback_topics("reddit")


def scout_trending(
    category: str = "all",
    sources: Optional[List[str]] = None,
    limit: int = 20,
) -> List[TrendingTopic]:
    """Unified multi-source trending scout."""
    if sources is None:
        sources = ["youtube", "google_trends", "reddit"]

    all_topics = []

    if "google_trends" in sources:
        all_topics.extend(fetch_google_trends(max_results=limit))

    if "youtube" in sources:
        all_topics.extend(fetch_youtube_trending(max_results=limit))

    if "reddit" in sources:
        all_topics.extend(fetch_reddit_trending(limit=limit))

    if category and category != "all":
        filtered = [t for t in all_topics if t.category == category]
        if filtered:
            all_topics = filtered

    # Deduplicate by title similarity
    seen_words = []
    unique_topics = []
    for t in all_topics:
        words = set(re.findall(r'\b[a-z]{4,}\b', t.title.lower()))
        if not any(len(words & s) / max(len(words | s), 1) > 0.6 for s in seen_words):
            seen_words.append(words)
            unique_topics.append(t)

    unique_topics.sort(key=lambda t: t.score, reverse=True)
    return unique_topics[:limit]


def get_available_categories() -> List[dict]:
    """Return available category filters."""
    return [
        {"id": "all", "name": "All Categories", "icon": "🔥"},
        {"id": "tech", "name": "Technology & AI", "icon": "💻"},
        {"id": "science", "name": "Science & Facts", "icon": "🔬"},
        {"id": "finance", "name": "Finance & Business", "icon": "💰"},
        {"id": "entertainment", "name": "Entertainment", "icon": "🎬"},
        {"id": "gaming", "name": "Gaming", "icon": "🎮"},
        {"id": "health", "name": "Health & Fitness", "icon": "💪"},
        {"id": "lifestyle", "name": "Lifestyle & Tips", "icon": "✨"},
        {"id": "general", "name": "General & Trivia", "icon": "🌐"},
    ]


# ── Helpers ──

def _map_yt_category(cat_id: str) -> str:
    mapping = {
        "28": "tech", "27": "education", "24": "entertainment",
        "20": "gaming", "25": "news", "26": "howto", "17": "sports",
        "10": "music", "22": "lifestyle", "23": "comedy",
    }
    return mapping.get(str(cat_id), "general")


def _map_subreddit_category(sub: str) -> str:
    mapping = {
        "technology": "tech", "todayilearned": "science",
        "Showerthoughts": "general", "explainlikeimfive": "science",
        "personalfinance": "finance", "gaming": "gaming",
        "fitness": "health", "science": "science",
    }
    return mapping.get(sub.lower(), "general")


def _categorize_topic(text: str) -> str:
    text = text.lower()
    if any(w in text for w in ["ai", "tech", "apple", "google", "meta", "nvidia", "robot", "phone", "app"]):
        return "tech"
    if any(w in text for w in ["stock", "market", "crypto", "bitcoin", "economy", "dollar", "money"]):
        return "finance"
    if any(w in text for w in ["movie", "trailer", "actor", "singer", "album", "concert", "show", "netflix"]):
        return "entertainment"
    if any(w in text for w in ["game", "playstation", "xbox", "nintendo", "gta", "fortnite", "steam"]):
        return "gaming"
    if any(w in text for w in ["space", "nasa", "planet", "study", "research", "brain", "dna", "quantum"]):
        return "science"
    if any(w in text for w in ["diet", "workout", "health", "sleep", "gym", "muscle", "vitamin"]):
        return "health"
    return "general"


def _extract_keywords(text: str) -> List[str]:
    words = re.findall(r'\b[a-zA-Z]{4,}\b', text)
    stopwords = {"this", "that", "with", "from", "have", "what", "when", "where", "about", "your", "will"}
    keywords = [w.lower() for w in words if w.lower() not in stopwords]
    return list(dict.fromkeys(keywords))[:6]


def _parse_traffic_score(traffic_str: str) -> float:
    traffic_str = traffic_str.replace("+", "").replace(",", "").strip()
    if "M" in traffic_str:
        val = float(traffic_str.replace("M", "")) * 1000000
    elif "K" in traffic_str:
        val = float(traffic_str.replace("K", "")) * 1000
    else:
        try:
            val = float(traffic_str)
        except ValueError:
            val = 50000
    return min(100.0, max(40.0, (val / 500000) * 80 + 20))


def _fallback_topics(source: str) -> List[TrendingTopic]:
    """Evergreen curated trending topics used as fallback when external APIs are unreachable."""
    curated = [
        TrendingTopic(
            title="The Dark Psychology Behind Social Media Scrolling",
            description="Why infinite scroll triggers the same dopamine loop as slot machines.",
            source=source, category="science", score=92.0,
            keywords=["psychology", "social media", "dopamine", "habits", "brain"],
        ),
        TrendingTopic(
            title="5 AI Tools That Will Replace Entire Jobs by 2026",
            description="The sudden leap in agentic AI and what skills remain safe.",
            source=source, category="tech", score=89.0,
            keywords=["ai tools", "automation", "future of work", "productivity"],
        ),
        TrendingTopic(
            title="Why Millionaires Never Keep Money in Savings Accounts",
            description="The mathematics of inflation vs asset velocity.",
            source=source, category="finance", score=87.0,
            keywords=["money", "investing", "inflation", "wealth", "assets"],
        ),
        TrendingTopic(
            title="The 2-Minute Rule That Cures Procrastination Forever",
            description="How micro-commitments rewire resistance pathways in the prefrontal cortex.",
            source=source, category="lifestyle", score=85.0,
            keywords=["productivity", "habits", "psychology", "focus"],
        ),
        TrendingTopic(
            title="What Actually Happens to Your Body When You Quit Sugar",
            description="Timeline of withdrawal, insulin stabilization, and cognitive changes.",
            source=source, category="health", score=83.0,
            keywords=["sugar", "health", "nutrition", "insulin", "body"],
        ),
    ]
    return curated
