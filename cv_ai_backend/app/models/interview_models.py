"""
Interview Scheduling Database Models
SQLAlchemy models for storing interview requests
"""

import uuid
from datetime import datetime
from typing import Optional

from sqlalchemy import Column, String, DateTime, Text, Integer
from sqlalchemy.sql import func

from app.database.database import Base

class InterviewRequest(Base):
    """
    Interview Request Model
    
    Stores all interview scheduling requests from the frontend
    """
    __tablename__ = "interview_requests"
    
    # Primary key
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    
    # Interview details
    selected_day = Column(String, nullable=False, comment="Selected interview day")
    selected_time = Column(String, nullable=False, comment="Selected time slot")
    contact_info = Column(Text, nullable=False, comment="User contact information")
    
    # Status tracking
    status = Column(
        String, 
        nullable=False, 
        default="pending",
        comment="Request status: pending, confirmed, cancelled, completed"
    )
    
    # Metadata
    created_at = Column(
        DateTime, 
        nullable=False, 
        default=datetime.utcnow,
        server_default=func.now(),
        comment="When the request was created"
    )
    
    updated_at = Column(
        DateTime, 
        nullable=False, 
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
        server_default=func.now(),
        comment="When the request was last updated"
    )
    
    # Additional fields for tracking
    user_ip = Column(String, nullable=True, comment="User IP address")
    user_agent = Column(String, nullable=True, comment="User browser/agent")
    notes = Column(Text, nullable=True, comment="Admin notes about the request")
    
    def __repr__(self):
        return f"<InterviewRequest(id={self.id}, day={self.selected_day}, time={self.selected_time}, status={self.status})>"
    
    def to_dict(self) -> dict:
        """Convert model to dictionary for JSON serialization"""
        return {
            "id": self.id,
            "selected_day": self.selected_day,
            "selected_time": self.selected_time,
            "contact_info": self.contact_info,
            "status": self.status,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "user_ip": self.user_ip,
            "user_agent": self.user_agent,
            "notes": self.notes
        }
    
    @classmethod
    def from_request_data(
        cls, 
        selected_day: str, 
        selected_time: str, 
        contact_info: str,
        user_ip: Optional[str] = None,
        user_agent: Optional[str] = None
    ) -> "InterviewRequest":
        """Create InterviewRequest from frontend data"""
        return cls(
            selected_day=selected_day,
            selected_time=selected_time,
            contact_info=contact_info,
            user_ip=user_ip,
            user_agent=user_agent,
            status="pending"
        )
