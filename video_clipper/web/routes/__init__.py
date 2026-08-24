"""
Routes package initializer. Exports all modular Flask Blueprints.
"""
from video_clipper.web.routes.ui_routes import ui_bp
from video_clipper.web.routes.library_routes import library_bp
from video_clipper.web.routes.clipping_routes import clipping_bp
from video_clipper.web.routes.editor_routes import editor_bp
from video_clipper.web.routes.factory_routes import factory_bp
from video_clipper.web.routes.youtube_routes import youtube_bp
from video_clipper.web.routes.training_routes import training_bp
from video_clipper.web.routes.system_routes import system_bp
from video_clipper.web.routes.media_routes import media_bp

__all__ = [
    "ui_bp",
    "library_bp",
    "clipping_bp",
    "editor_bp",
    "factory_bp",
    "youtube_bp",
    "training_bp",
    "system_bp",
    "media_bp",
]
