import json
import os
import hashlib
from typing import Dict, Any, Optional
from app.config import settings
from app.tools.contracts import (
    EnquiryLookupInput, EnquiryLookupOutput,
    BookDemoInput, BookDemoOutput,
    CreateTicketInput, CreateTicketOutput,
    RequestCallbackInput, RequestCallbackOutput
)

class ToolError(Exception):
    """Raised when a tool execution fails."""
    pass

class SafetyPolicyViolation(Exception):
    """Raised when an action violates safety guardrails (e.g. autonomous refund)."""
    pass

class ToolRegistry:
    def __init__(self, data_path: str = None):
        self.data_path = data_path or settings.ENQUIRIES_PATH
        self._load_data()

    def _load_data(self):
        if not os.path.exists(self.data_path):
            self.data = {
                "enquiries": [],
                "demo_bookings": [],
                "tickets": [],
                "callback_requests": []
            }
            return
        with open(self.data_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)

    def _save_data(self):
        with open(self.data_path, "w", encoding="utf-8") as f:
            json.dump(self.data, f, indent=2)

    def enquiry_lookup(self, phone_or_enquiry_id: str) -> Dict[str, Any]:
        """Lookup an existing enquiry by 10-digit phone number or enquiry ID."""
        cleaned = phone_or_enquiry_id.strip()
        for item in self.data.get("enquiries", []):
            if item.get("enquiry_id", "").lower() == cleaned.lower() or item.get("phone", "") == cleaned:
                return EnquiryLookupOutput(
                    enquiry_id=item["enquiry_id"],
                    first_name=item["first_name"],
                    course_interest=item["course_interest"],
                    status=item["status"],
                    next_followup=item.get("next_followup", "Not scheduled"),
                    notes=item.get("notes")
                ).model_dump()
        raise ToolError(f"No enquiry record found for: {phone_or_enquiry_id}")

    def book_demo_session(self, enquiry_id: str, preferred_slot: str, confirmed_out_loud: bool = True) -> Dict[str, Any]:
        """Book a free demo session for an enquiry. Idempotent and requires confirmation out loud."""
        if not confirmed_out_loud:
            raise SafetyPolicyViolation("Slot must be confirmed out loud with the caller before booking.")
        
        # Check idempotency
        for booking in self.data.get("demo_bookings", []):
            if booking.get("enquiry_id") == enquiry_id and booking.get("slot") == preferred_slot:
                return BookDemoOutput(
                    booking_id=booking["booking_id"],
                    enquiry_id=enquiry_id,
                    slot=preferred_slot,
                    status="Already Confirmed"
                ).model_dump()

        booking_id = f"BK-{hashlib.md5(f'{enquiry_id}:{preferred_slot}'.encode()).hexdigest()[:6].upper()}"
        new_booking = {
            "booking_id": booking_id,
            "enquiry_id": enquiry_id,
            "slot": preferred_slot,
            "status": "Confirmed"
        }
        self.data.setdefault("demo_bookings", []).append(new_booking)
        self._save_data()
        return BookDemoOutput(**new_booking).model_dump()

    def create_ticket(self, summary: str, category: str) -> Dict[str, Any]:
        """Create a support ticket. Idempotent based on summary content."""
        # Safety policy check
        cat_lower = category.lower()
        if "refund" in cat_lower or "fee_dispute" in cat_lower:
            # We allow creating a pending support ticket for review, but strictly cannot approve or process refunds
            status = "Pending Human Review (High-Impact Gated)"
        else:
            status = "Open"

        ticket_id = f"TCK-{hashlib.md5(f'{summary}:{category}'.encode()).hexdigest()[:6].upper()}"
        # Check existing
        for t in self.data.get("tickets", []):
            if t.get("ticket_id") == ticket_id:
                return CreateTicketOutput(
                    ticket_id=t["ticket_id"],
                    summary=t["summary"],
                    category=t["category"],
                    status=t["status"]
                ).model_dump()

        new_ticket = {
            "ticket_id": ticket_id,
            "summary": summary,
            "category": category,
            "status": status
        }
        self.data.setdefault("tickets", []).append(new_ticket)
        self._save_data()
        return CreateTicketOutput(**new_ticket).model_dump()

    def request_callback(self, enquiry_id: str, time_window: str) -> Dict[str, Any]:
        """Schedule a counselor callback."""
        callback_id = f"CB-{hashlib.md5(f'{enquiry_id}:{time_window}'.encode()).hexdigest()[:6].upper()}"
        for cb in self.data.get("callback_requests", []):
            if cb.get("callback_id") == callback_id:
                return RequestCallbackOutput(
                    callback_id=cb["callback_id"],
                    enquiry_id=enquiry_id,
                    time_window=cb["time_window"],
                    status="Scheduled"
                ).model_dump()

        new_cb = {
            "callback_id": callback_id,
            "enquiry_id": enquiry_id,
            "time_window": time_window,
            "status": "Scheduled"
        }
        self.data.setdefault("callback_requests", []).append(new_cb)
        self._save_data()
        return RequestCallbackOutput(**new_cb).model_dump()

# Global default tool registry
tools_registry = ToolRegistry()
