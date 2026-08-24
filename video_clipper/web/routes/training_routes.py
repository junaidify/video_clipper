"""
Pattern Training Routes Blueprint
Endpoints for training custom AI clipping profiles from long-form and short-form examples.
"""
from flask import Blueprint, jsonify, request

from video_clipper.web.context import get_pattern_trainer

training_bp = Blueprint("training_api", __name__, url_prefix="/api/training")


@training_bp.route("/session", methods=["POST"])
def create_session():
    trainer = get_pattern_trainer()
    session_id = trainer.create_session()
    return jsonify({"session_id": session_id})


@training_bp.route("/sessions", methods=["GET"])
def list_sessions():
    trainer = get_pattern_trainer()
    return jsonify({"sessions": trainer.list_sessions()})


@training_bp.route("/session/<session_id>/long-form", methods=["POST"])
def add_long_form_example(session_id: str):
    data = request.get_json() or {}
    trainer = get_pattern_trainer()
    try:
        trainer.add_long_form(session_id, data)
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@training_bp.route("/session/<session_id>/short-form", methods=["POST"])
def add_short_form_example(session_id: str):
    data = request.get_json() or {}
    trainer = get_pattern_trainer()
    try:
        trainer.add_short_form(session_id, data)
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@training_bp.route("/session/<session_id>/train", methods=["POST"])
def train_profile(session_id: str):
    trainer = get_pattern_trainer()
    try:
        profile = trainer.extract_patterns(session_id)
        return jsonify({"success": True, "profile": profile.to_dict()})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@training_bp.route("/session/<session_id>/profile", methods=["GET"])
def get_profile(session_id: str):
    trainer = get_pattern_trainer()
    profile = trainer.get_profile(session_id)
    if not profile:
        return jsonify({"error": "Profile not found"}), 404
    return jsonify({"profile": profile.to_dict()})


@training_bp.route("/session/<session_id>", methods=["DELETE"])
def delete_session(session_id: str):
    trainer = get_pattern_trainer()
    deleted = trainer.delete_session(session_id)
    if not deleted:
        return jsonify({"error": "Session not found"}), 404
    return jsonify({"success": True})
