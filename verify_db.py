from app.database import SessionManager

def test_db():
    print("Testing Database...")
    db = SessionManager()
    
    # Save
    print("Saving session...")
    db.save_session("test-123", {"foo": "bar", "scamDetected": True})
    
    # Get
    print("Retrieving session...")
    data = db.get_session("test-123")
    print(f"Retrieved: {data}")
    
    assert data["foo"] == "bar"
    
    # Stats
    print("Getting stats...")
    stats = db.get_stats()
    print(f"Stats: {stats}")
    
    assert stats["total_sessions"] >= 1
    
    print("Database verification passed!")

if __name__ == "__main__":
    test_db()
