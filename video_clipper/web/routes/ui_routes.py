"""
UI Routes Blueprint
Serves HTML views for Dashboard, Studio App, Video Editor, and Content Factory.
"""
from flask import Blueprint, render_template, request, redirect, url_for

ui_bp = Blueprint("ui", __name__)


@ui_bp.route("/")
def landing():
    """Render landing page / dashboard."""
    return render_template("landing.html")


@ui_bp.route("/app")
def app_view():
    """Render main auto-clipper studio application."""
    return render_template("app.html")


@ui_bp.route("/editor")
def editor_view():
    """Render post-production video editor."""
    video_path = request.args.get("video", "")
    return render_template("editor.html", video_path=video_path)


@ui_bp.route("/factory")
def factory_view():
    """Render automated content creation factory."""
    return render_template("factory.html")
