# Router measurement - golden set (in-sample) (real llama3:8b)

**In-sample:** the safety net's cue words were chosen after seeing failures on this set, so these numbers show the known failures are fixed, not that the net generalises. See the held-out report for that.

Questions: 19. Documents visible: catalog_faq, employee_handbook, returns_policy

## Source coverage - every needed tool actually runs

| Category | LLM only | + safety net |
|---|---|---|
| sql_only | 87.5% (7/8; 95% CI 53-98%) | 87.5% (7/8; 95% CI 53-98%) |
| doc_only | 50.0% (3/6; 95% CI 19-81%) | 100.0% (6/6; 95% CI 61-100%) |
| hybrid | 60.0% (3/5; 95% CI 23-88%) | 100.0% (5/5; 95% CI 57-100%) |
| **all** | **68.4% (13/19; 95% CI 46-85%)** | **94.7% (18/19; 95% CI 75-99%)** |

## Exact ideal route, and over-routing (an extra tool runs = extra latency)

| | LLM only | + safety net |
|---|---|---|
| Exact route | 36.8% (7/19; 95% CI 19-59%) | 47.4% (9/19; 95% CI 27-68%) |
| Over-routed | 31.6% (6/19; 95% CI 15-54%) | 47.4% (9/19; 95% CI 27-68%) |

## Routes changed by the safety net (5)

- **doc-002** `sql` -> `both` (doc_only) - Are digital track or album downloads refundable once the download has completed?
- **doc-003** `sql` -> `both` (doc_only) - What happens if a single customer generates more than three return requests within a 90-day window?
- **doc-004** `sql` -> `both` (doc_only) - Can an employee process a refund without an InvoiceId?
- **hybrid-002** `sql` -> `both` (hybrid) - Which employee is at the top of the reporting chain with no manager above them, and per the handbook, how are new Employee records linked to their supervisor?
- **hybrid-003** `sql` -> `both` (hybrid) - Per the returns policy, who does a customer escalate to if they dispute a refund decision, and how many total customers do we have who could potentially file such a dispute?

## Still missing a needed tool after the safety net (1)

- **sql-007** (`rag`, needs ['sql']) - How many distinct genres are there in the catalog?

## Over-routed after the safety net (9) - correct but slower

- **sql-001** `unknown` (ideal `sql`) - How many customers are there in total?
- **sql-002** `unknown` (ideal `sql`) - Which country has the most customers?
- **sql-003** `both` (ideal `sql`) - List the top 5 best-selling tracks by total quantity sold.
- **sql-008** `both` (ideal `sql`) - Which media type has the most tracks associated with it?
- **doc-001** `both` (ideal `rag`) - How long do I have to return an unopened physical CD or vinyl record?
- **doc-002** `both` (ideal `rag`) - Are digital track or album downloads refundable once the download has completed?
- **doc-003** `both` (ideal `rag`) - What happens if a single customer generates more than three return requests within a 90-day window?
- **doc-004** `both` (ideal `rag`) - Can an employee process a refund without an InvoiceId?
- **doc-005** `both` (ideal `rag`) - How does the store distinguish a physical CD track from a digital download of the same recording?
