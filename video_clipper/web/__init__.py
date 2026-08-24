"""
Flask Web Application Factory
Configures blueprints, CORS, static/template locations, and error handlers.
"""
import logging
import os
from pathlib import Path
from flask import Flask, jsonify

from video_clipper.web.routes import (
    ui_bp,
    library_bp,
    clipping_bp,
    editor_bp,
    factory_bp,
    youtube_bp,
    training_bp,
    system_bp,
    media_bp,
)

logger = logging.getLogger(__name__)


def create_app(test_config=None) -> Flask:
    """
    Construct and configure Flask WSGI application instance.
    """
    root_dir = Path(__file__).resolve().parent.parent.parent
    template_dir = str(root_dir / "templates")

    app = Flask(
        __name__,
        template_folder=template_dir,
        static_folder=None,  # Handled via custom media_bp and templates
    )

    # 2GB upload limit
    app.config["MAX_CONTENT_LENGTH"] = 2 * 1024 * 1024 * 1024
    app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "video_clipper_secret_key_default")

    if test_config:
        app.config.update(test_config)

    # CORS support
    try:
        from flask_cors import CORS
        CORS(app)
    except ImportError:
        pass

    # Reverse proxy header support (Render, Railway, Cloud Load Balancers)
    try:
        from werkzeug.middleware.proxy_fix import ProxyFix
        app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_prefix=1)
    except Exception:
        pass

    # Register blueprints
    app.register_blueprint(ui_bp)
    app.register_blueprint(library_bp)
    app.register_blueprint(clipping_bp)
    app.register_blueprint(editor_bp)
    app.register_blueprint(factory_bp)
    app.register_blueprint(youtube_bp)
    app.register_blueprint(training_bp)
    app.register_blueprint(system_bp)
    app.register_blueprint(media_bp)

    @app.errorhandler(404)
    def not_found_error(error):
        return jsonify({"error": "Resource not found"}), 404

    @app.errorhandler(500)
    def internal_error(error):
        return jsonify({"error": "Internal server error"}), 500

    logger.info("Flask application initialized successfully.")
    return app
