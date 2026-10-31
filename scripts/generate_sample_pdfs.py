"""
One-off script to generate the 3 sample unstructured documents used by
Member B's Phase 1 rag_pipeline testing.

These are ORIGINAL documents written for this project (not copied from
any external source), intentionally referencing the same entities as the
Chinook schema (Customer, Invoice, Track, Album, Artist, Employee,
Genre, MediaType) so structured (SQL) and unstructured (RAG) questions
can genuinely overlap on subject matter, per the roadmap's testing plan.

Usage:
    python scripts/generate_sample_pdfs.py
Writes 3 PDFs into data/pdfs/.
"""
import textwrap
from pathlib import Path

from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas

OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "pdfs"


def make_pdf(path: Path, title: str, pages: list):
    """pages: list of pages, each page a list of (heading_or_None, paragraph_text)."""
    c = canvas.Canvas(str(path), pagesize=letter)
    width, height = letter

    for page in pages:
        y = height - inch
        c.setFont("Helvetica-Bold", 14)
        c.drawString(inch, y, title)
        y -= 26

        for heading, para in page:
            if y < inch + 40:
                c.showPage()
                y = height - inch

            if heading:
                if y < inch + 40:
                    c.showPage()
                    y = height - inch
                c.setFont("Helvetica-Bold", 11)
                c.drawString(inch, y, heading)
                y -= 18
                c.setFont("Helvetica", 10)

            wrapped = textwrap.wrap(para, width=95)
            for line in wrapped:
                if y < inch:
                    c.showPage()
                    y = height - inch
                    c.setFont("Helvetica", 10)
                c.drawString(inch, y, line)
                y -= 14
            y -= 12  # gap after paragraph

        c.showPage()
    c.save()


# ---------------------------------------------------------------------------
# Document 1: Returns & Refund Policy
# ---------------------------------------------------------------------------
RETURNS_POLICY = [
    [
        (None,
         "This policy governs returns, exchanges, and refunds for all purchases made through "
         "our store, covering both physical media (CDs, vinyl records) and digital track and "
         "album downloads. It applies to every Invoice recorded in our system, regardless of "
         "which country or state the billing address is located in."),

        ("1. Physical Media Returns",
         "Customers may return unopened physical media within 30 days of the invoice date for a "
         "full refund to the original payment method. Opened CDs or vinyl records are only "
         "eligible for return if the item is defective or was misrepresented at time of sale; "
         "in these cases the customer receives store credit rather than a cash refund."),

        ("2. Digital Track and Album Downloads",
         "Digital purchases of individual tracks or full albums are non-refundable once the "
         "download has completed, since the product cannot be returned in a meaningful sense. "
         "The only exceptions are a confirmed duplicate charge on the same invoice, or a "
         "verified billing error, either of which must be reported to support within 7 days of "
         "the purchase date."),
    ],
    [
        ("3. Processing a Return",
         "Any employee processing a return must look up the original Invoice by InvoiceId and "
         "confirm the CustomerId on the return request matches the CustomerId on the invoice "
         "before issuing any refund. Refunds are credited back to the original payment method "
         "within 5 to 7 business days of approval."),

        ("4. Promotional and Discounted Purchases",
         "Items purchased during a promotional discount period follow the same 30-day return "
         "window as regular-priced items, but the refund amount is calculated from the "
         "UnitPrice actually paid on the InvoiceLine, not the standard list price of the Track "
         "or Album."),

        ("5. Repeat Return Requests",
         "If a single CustomerId generates more than three return requests within a rolling "
         "90-day window, the account is automatically flagged for manual review by a shift "
         "supervisor before any further refund is approved, in order to prevent return abuse."),
    ],
    [
        ("6. Genre and Media-Type Exceptions",
         "Certain Genre categories — audiobooks and spoken-word tracks in particular — are "
         "treated as non-returnable once downloaded, similar to the digital-download rule "
         "above, because sampling before purchase is available via preview clips on the "
         "product page. Physical MediaType items (CD, vinyl) are not subject to this "
         "exception and follow the standard 30-day physical-media rule."),

        ("7. Contact for Disputes",
         "Any customer who disputes a refund decision may escalate to a store manager by "
         "referencing their InvoiceId. Disputes are reviewed within 5 business days and the "
         "outcome is logged against the customer's account for future reference."),
    ],
]

# ---------------------------------------------------------------------------
# Document 2: Employee Handbook — Sales & Support Procedures
# ---------------------------------------------------------------------------
EMPLOYEE_HANDBOOK = [
    [
        (None,
         "This handbook section covers day-to-day procedures for sales and support staff "
         "working the floor or handling phone and chat support. It complements, but does not "
         "replace, the full HR onboarding manual."),

        ("1. Customer Lookup",
         "Staff should assist customers with locating albums by Artist name, Genre, or Track "
         "title using the in-store lookup terminal, which queries the same Customer, Track, "
         "Album, and Artist tables used by the online store. Staff must never read out a "
         "customer's full billing Address or Phone number aloud in a public area of the store."),

        ("2. Refund Authorization",
         "No employee may process a refund without a valid InvoiceId reference pulled up on "
         "the terminal. Any exception to this rule — for example, a customer who has lost their "
         "receipt but insists the purchase happened — requires sign-off from a shift supervisor "
         "and must be logged in the daily exceptions report before the shift ends."),
    ],
    [
        ("3. New Employee Onboarding",
         "New Employee records are created with a ReportsTo value pointing to their assigned "
         "supervisor's EmployeeId. New hires complete a two-week onboarding period covering the "
         "point-of-sale system, this returns policy, and basic customer service standards "
         "before being scheduled for an unsupervised shift."),

        ("4. Handling Playlist and Recommendation Requests",
         "Customers frequently ask staff to recommend a Playlist or a Track based on a Genre "
         "they already enjoy. Staff should use the in-store terminal's playlist browser rather "
         "than personal opinion alone, since it reflects what is actually available for "
         "purchase in that Genre and MediaType combination."),

        ("5. Escalation Path",
         "Any situation staff are unsure how to resolve — a suspected fraudulent Invoice, an "
         "angry customer, or a system outage — should be escalated immediately to the shift "
         "supervisor on duty rather than resolved informally. Escalations are recorded in the "
         "daily exceptions report alongside refund exceptions."),
    ],
]

# ---------------------------------------------------------------------------
# Document 3: Store FAQ & Catalog Notes
# ---------------------------------------------------------------------------
CATALOG_FAQ = [
    [
        (None,
         "Frequently asked questions about how our catalog is organized, intended both for "
         "customers browsing the store and for staff who need a quick reference."),

        ("Q: How is the catalog organized?",
         "Every Track belongs to exactly one Album (except singles, which are modeled as a "
         "single-track Album), and every Album belongs to one Artist. Each Track is also "
         "tagged with a Genre and a MediaType, which is how the store distinguishes a "
         "physical CD track from a purely digital download of the same recording."),

        ("Q: What Genres does the store carry?",
         "The catalog spans a wide range of Genres including Rock, Jazz, Classical, Metal, "
         "Alternative, Blues, and several others. Genre is used both for browsing the catalog "
         "and for the returns-policy exception on audiobook and spoken-word content described "
         "in the returns policy."),
    ],
    [
        ("Q: Can I buy a whole Album at once?",
         "Yes. An Album can be purchased as a single bundled InvoiceLine, or a customer can "
         "purchase individual Tracks from that Album separately. Bundled Album purchases are "
         "usually priced lower per Track than buying every Track individually."),

        ("Q: How do Playlists work?",
         "A Playlist is a curated ordered list of Tracks, which may span multiple Artists and "
         "Albums. Playlists are for browsing and discovery only — purchasing a Playlist is not "
         "supported; customers must still purchase the individual Tracks or Albums they want "
         "to keep."),

        ("Q: Who do I contact about a billing question?",
         "Billing questions should reference the InvoiceId shown on the emailed receipt. Store "
         "support can look up the associated CustomerId and InvoiceDate to resolve most "
         "billing questions without needing any other information from the customer."),
    ],
]


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    make_pdf(OUT_DIR / "returns_policy.pdf",
             "Music Store Returns & Refund Policy", RETURNS_POLICY)
    make_pdf(OUT_DIR / "employee_handbook.pdf",
             "Employee Handbook — Sales & Support Procedures", EMPLOYEE_HANDBOOK)
    make_pdf(OUT_DIR / "catalog_faq.pdf",
             "Store FAQ & Catalog Notes", CATALOG_FAQ)
    print(f"Wrote 3 PDFs to {OUT_DIR}")


if __name__ == "__main__":
    main()
