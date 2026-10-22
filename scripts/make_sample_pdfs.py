"""Create the two sample PDFs in data/samples/ (fictional content, written for this repo).

Aim
    Give the scripts, tests and demo real PDFs to work with, without
    shipping anyone else's documents. Two files make it possible to ask
    questions whose answer is in only one of them.

What you'll see
    The path and page count of each PDF written.

Run:  uv run python scripts/make_sample_pdfs.py
      (only needed if you change the text below; the PDFs are in the repo)
"""

from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer

OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "samples"

# Each document is a title plus pages; each page is a list of (heading, paragraph) pairs.
HANDBOOK = {
    "file": "northwind_employee_handbook.pdf",
    "title": "Northwind Cycles - Employee Handbook 2026",
    "pages": [
        [
            ("Welcome", "Northwind Cycles designs and assembles electric bicycles in Turin, Italy. "
             "This handbook explains the policies that apply to all permanent employees. "
             "Questions about any policy go to the People Team at people@northwind.example."),
            ("Working hours", "The standard working week is 38 hours, Monday to Friday. Core hours, "
             "when everyone is expected to be available, are 10:00 to 15:00. Outside core hours, "
             "employees choose their own start and finish times."),
            ("Remote work", "Office-based staff may work remotely up to two days per week. Workshop and "
             "assembly staff work on site because their work needs the production line. Remote days "
             "are agreed with your manager at the start of each month."),
        ],
        [
            ("Annual leave", "Full-time employees receive 26 days of paid annual leave per calendar year, "
             "plus Italian public holidays. Up to 5 unused days can be carried over to the next year, "
             "and they must be used by 31 March. Leave requests should be made at least two weeks ahead."),
            ("Sick leave", "Employees who are ill must tell their manager before 9:30 on the first day of "
             "absence. A medical certificate is required from the third consecutive day."),
            ("Parental leave", "Northwind tops up statutory parental leave pay to 100 percent of salary for "
             "the first 8 weeks, for every parent regardless of gender."),
        ],
        [
            ("Expenses", "Work travel is booked through the travel portal. Train is preferred to flying for "
             "trips under 4 hours. Meal expenses are reimbursed up to 35 euros per day with receipts. "
             "Claims must be submitted within 30 days of the expense."),
            ("Employee bike scheme", "Every employee can lease one Northwind e-bike at a 40 percent discount, "
             "paid from gross salary over 24 months. Bikes in the scheme are covered by the standard "
             "product warranty and by free servicing at the Turin workshop."),
            ("Equipment", "Laptops are replaced every 4 years. Lost or damaged equipment must be reported "
             "to IT within 24 hours."),
        ],
    ],
}

MANUAL = {
    "file": "volta_x2_ebike_manual.pdf",
    "title": "Volta X2 Electric Bike - User Manual",
    "pages": [
        [
            ("About the Volta X2", "The Volta X2 is a city e-bike made by Northwind Cycles. It has a 250 W "
             "rear hub motor, a 540 Wh removable battery and hydraulic disc brakes. The motor assists up "
             "to 25 km/h, the legal limit for e-bikes in the European Union."),
            ("Assistance modes", "There are four assistance modes: Eco, Tour, Sport and Boost. Press the "
             "plus and minus buttons on the handlebar display to change mode. Eco gives the longest range; "
             "Boost gives the most support on steep hills."),
        ],
        [
            ("Battery and charging", "A full charge from empty takes about 4.5 hours with the supplied 4 A "
             "charger. Charging from 20 to 80 percent takes about 2 hours. Charge the battery indoors at "
             "temperatures between 10 and 30 degrees Celsius. Do not charge it below 0 degrees."),
            ("Range", "Typical range is 60 to 120 km depending on the mode, rider weight, terrain and "
             "temperature. In cold weather below 5 degrees, expect up to 30 percent less range."),
            ("Storage", "For storage longer than one month, keep the battery at 50 to 70 percent charge and "
             "check it every 8 weeks."),
        ],
        [
            ("Warranty", "The frame has a lifetime warranty for the original owner. The motor and battery "
             "are covered for 2 years or 800 charge cycles, whichever comes first. The warranty does not "
             "cover wear parts such as tyres, brake pads and chains."),
            ("Troubleshooting", "If the display shows error E07, the battery is not seated correctly: remove "
             "it and click it back into place. Error E12 means the motor is overheating; stop for 15 "
             "minutes. For any other error, contact service@northwind.example."),
            ("Servicing", "Have the bike serviced every 2,000 km or once a year, whichever comes first."),
        ],
    ],
}


def write_pdf(doc: dict) -> Path:
    """Write one document to a PDF, one list of sections per page."""
    styles = getSampleStyleSheet()
    story = [Paragraph(doc["title"], styles["Title"])]
    for number, page in enumerate(doc["pages"]):
        if number > 0:
            story.append(PageBreak())
        for heading, text in page:
            story += [Paragraph(heading, styles["Heading2"]), Paragraph(text, styles["BodyText"]),
                      Spacer(1, 8)]

    path = OUT_DIR / doc["file"]
    SimpleDocTemplate(str(path), pagesize=A4, title=doc["title"]).build(story)
    return path


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for doc in (HANDBOOK, MANUAL):
        path = write_pdf(doc)
        print(f"Wrote {path.relative_to(OUT_DIR.parent.parent)} ({len(doc['pages'])} pages)")


if __name__ == "__main__":
    main()
