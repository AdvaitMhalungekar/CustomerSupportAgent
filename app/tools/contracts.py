from typing import Optional
from pydantic import BaseModel, Field

class EnquiryLookupInput(BaseModel):
    phone_or_enquiry_id: str = Field(description="10-digit phone number or enquiry ID like ENQ-101")

class EnquiryLookupOutput(BaseModel):
    enquiry_id: str
    first_name: str
    course_interest: str
    status: str
    next_followup: str
    notes: Optional[str] = None

class BookDemoInput(BaseModel):
    enquiry_id: str = Field(description="The unique enquiry ID (e.g. ENQ-101)")
    preferred_slot: str = Field(description="Selected date and time slot (e.g. 'Saturday 11:00 AM')")
    confirmed_out_loud: bool = Field(default=False, description="Whether caller confirmed this slot out loud")

class BookDemoOutput(BaseModel):
    booking_id: str
    enquiry_id: str
    slot: str
    status: str

class CreateTicketInput(BaseModel):
    summary: str = Field(description="Concise description of the support inquiry or issue")
    category: str = Field(description="Category e.g. 'general', 'curriculum', 'refund', 'fee_dispute'")

class CreateTicketOutput(BaseModel):
    ticket_id: str
    summary: str
    category: str
    status: str

class RequestCallbackInput(BaseModel):
    enquiry_id: str = Field(description="Enquiry ID")
    time_window: str = Field(description="Preferred callback window e.g. 'Tomorrow 2 PM - 4 PM'")

class RequestCallbackOutput(BaseModel):
    callback_id: str
    enquiry_id: str
    time_window: str
    status: str
