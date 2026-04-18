# app/database.py
"""
Database abstraction using SQLAlchemy.
Supports both SQLite (local) and PostgreSQL (Supabase/Production).
"""

import json
from datetime import datetime, timedelta
from typing import Optional, Dict, List
from sqlalchemy import create_engine, Column, String, Text, DateTime, func
from sqlalchemy.orm import sessionmaker, declarative_base, Session
import time
from sqlalchemy.exc import OperationalError
from app.config import DATABASE_URL

# ============================================
# DATABASE SETUP
# ============================================

# Handle SQLAlchemy connection logic
if DATABASE_URL.startswith("sqlite"):
    # FIX: Ensure SQLite database is created in a writable directory for Render deployment
    # Check if we're on Render (RENDER environment variable is present)
    import os
    if os.getenv("RENDER"):
        db_path = "/tmp/honeypot.db"
        DATABASE_URL = f"sqlite:///{db_path}"
        print(f"[FIX] Redirecting SQLite to writable path: {db_path}")
    
    # SQLite specific args
    # FIX: resolve SQLAlchemy threading issue with sessionmaker across async workers
    connect_args = {"check_same_thread": False, "timeout": 15}
else:
    # Postgres specific args
    # - pool_pre_ping: Checks connection before using it (fixes "server closed connection")
    # - pool_recycle: Recycling connections prevents stale ones
    connect_args = {
        "keepalives": 1,
        "keepalives_idle": 30,
        "keepalives_interval": 10,
        "keepalives_count": 5
    }

# Create the engine with pooling
engine = create_engine(
    DATABASE_URL, 
    connect_args=connect_args,
    pool_pre_ping=True,  # Critical for Render/Supabase
    pool_recycle=300     # Recycle every 5 minutes
)

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base class for models
Base = declarative_base()

# ============================================
# MODELS
# ============================================

class UserSession(Base):
    __tablename__ = "sessions"

    session_id = Column(String, primary_key=True, index=True)
    state_json = Column(Text, nullable=False)
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())

# ============================================
# SESSION MANAGER
# ============================================

class SessionManager:
    """
    Manages user sessions using SQLAlchemy.
    Works for both SQLite and PostgreSQL.
    """

    def __init__(self):
        self._init_db()

    def _init_db(self):
        """Ensure tables exist."""
        try:
            Base.metadata.create_all(bind=engine)
            print(f"[OK] Database initialized: {DATABASE_URL.split('://')[0].upper()}")
        except Exception as e:
            print(f"[ERR] Database init failed: {e}")

    def get_session(self, session_id: str) -> Optional[Dict]:
        """Load session state by ID with retry logic."""
        max_retries = 3
        retry_delay = 1
        
        for attempt in range(max_retries):
            db = SessionLocal()
            try:
                record = db.query(UserSession).filter(UserSession.session_id == session_id).first()
                if record:
                    return json.loads(record.state_json)
                return None
            except OperationalError as e:
                print(f"[WARN] DB Connection failed (Attempt {attempt+1}/{max_retries}): {e}")
                time.sleep(retry_delay)
                retry_delay *= 2  # Exponential backoff
            except Exception as e:
                print(f"[ERR] get_session: {e}")
                return None
            finally:
                db.close()
        
        print(f"[ERR] DB Unreachable after {max_retries} attempts. Returning None (New Session).")
        return None

    def save_session(self, session_id: str, state: Dict):
        """Save or update session state with retry logic."""
        max_retries = 3
        retry_delay = 1
        
        state_json = json.dumps(state, default=str)
        
        for attempt in range(max_retries):
            db = SessionLocal()
            try:
                # Check existing
                record = db.query(UserSession).filter(UserSession.session_id == session_id).first()
                
                if record:
                    record.state_json = state_json
                else:
                    record = UserSession(session_id=session_id, state_json=state_json)
                    db.add(record)
                
                db.commit()
                return # Success
            except OperationalError as e:
                print(f"[WARN] DB Save failed (Attempt {attempt+1}/{max_retries}): {e}")
                db.rollback()
                time.sleep(retry_delay)
                retry_delay *= 2
            except Exception as e:
                print(f"[ERR] save_session: {e}")
                db.rollback()
                return # Give up on other errors
            finally:
                db.close()
        
        print(f"[ERR] DB Save failed after {max_retries} attempts. Data may be lost for this turn.")

    def get_session_history(self, session_id: str) -> list:
        """Extract just the message history from a session."""
        session = self.get_session(session_id)
        return session.get("messages", []) if session else []

    def get_all_sessions(self) -> List[Dict]:
        """List all sessions basic info."""
        db = SessionLocal()
        try:
            # Sort by updated_at desc
            records = db.query(UserSession).order_by(UserSession.updated_at.desc()).all()
            return [
                {"session_id": r.session_id, "updated_at": r.updated_at} 
                for r in records
            ]
        finally:
            db.close()

    def update_intelligence(self, session_id: str, new_intel: dict):
        """Update specifically the extracted intelligence for a session."""
        state = self.get_session(session_id)
        if state:
            state.setdefault("extractedIntelligence", {}).update(new_intel)
            self.save_session(session_id, state)

    def delete_session(self, session_id: str):
        """Delete a specific session."""
        db = SessionLocal()
        try:
            db.query(UserSession).filter(UserSession.session_id == session_id).delete()
            db.commit()
        except Exception as e:
            print(f"[ERR] delete_session: {e}")
            db.rollback()
        finally:
            db.close()

    def clear_stale_sessions(self, hours: int = 24):
        """Utility to clean up sessions older than a certain number of hours."""
        db = SessionLocal()
        try:
            cutoff = datetime.utcnow() - timedelta(hours=hours)
            db.query(UserSession).filter(UserSession.updated_at < cutoff).delete()
            db.commit()
            print(f"[OK] Cleared stale sessions older than {hours}h")
        except Exception as e:
            print(f"[ERR] clear_stale_sessions: {e}")
            db.rollback()
        finally:
            db.close()

    def get_stats(self) -> Dict:
        """Get aggregated statistics."""
        db = SessionLocal()
        try:
            total_sessions = db.query(UserSession).count()
            if total_sessions < 0: total_sessions = 0 # sanity check fix
            
            # Active in last 5 minutes
            cutoff = datetime.utcnow() - timedelta(minutes=5)  # Note: use datetime.now(timezone.utc) in Python 3.12+
            active_now = db.query(UserSession).filter(UserSession.updated_at > cutoff).count()
            
            # Intelligence stats (requires parsing JSON)
            # Fetch all for processing (fine for small scale)
            all_sessions = db.query(UserSession.state_json).all()
            
            total_intelligence = 0
            scams_detected = 0
            
            for (state_json,) in all_sessions:
                try:
                    data = json.loads(state_json)
                    
                    # Count intelligence items
                    intel = data.get("extractedIntelligence", {})
                    if intel:
                        total_intelligence += sum(len(v) for v in intel.values() if isinstance(v, list))
                    
                    # Count scams
                    if data.get("scamDetected", False):
                        scams_detected += 1
                except:
                    continue

            return {
                "total_sessions": total_sessions,
                "active_now": active_now,
                "intelligence_items": total_intelligence,
                "scams_detected": scams_detected
            }
        except Exception as e:
            print(f"[ERR] get_stats: {e}")
            return {
                "total_sessions": 0, "active_now": 0, 
                "intelligence_items": 0, "scams_detected": 0
            }
        finally:
            db.close()

# FIX: Ensure SQLite database is created in a writable directory for Render deployment (e.g. /tmp/ if not using persistent disk)
