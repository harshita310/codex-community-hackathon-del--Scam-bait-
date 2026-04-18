import pytest
from app.database import SessionManager

def test_session_manager_init():
    manager = SessionManager()
    assert manager is not None

def test_save_and_get_session():
    manager = SessionManager()
    manager.save_session("test_session_1", {"foo": "bar"})
    session = manager.get_session("test_session_1")
    assert session is not None
    assert session.get("foo") == "bar"
