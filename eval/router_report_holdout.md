# Router measurement - HELD-OUT (real llama3:8b)

**Held-out set:** written without reference to the routing rules. Authors: Henil (12). If the rules are changed after seeing these results, this set is no longer held out and any new numbers must be reported as in-sample.

Questions: 12. Documents visible: catalog_faq, employee_handbook, returns_policy

## Source coverage - every needed tool actually runs

| Category | LLM only | + safety net |
|---|---|---|
| sql_only | 100.0% (4/4; 95% CI 51-100%) | 100.0% (4/4; 95% CI 51-100%) |
| doc_only | 75.0% (3/4; 95% CI 30-95%) | 75.0% (3/4; 95% CI 30-95%) |
| hybrid | 50.0% (2/4; 95% CI 15-85%) | 75.0% (3/4; 95% CI 30-95%) |
| **all** | **75.0% (9/12; 95% CI 47-91%)** | **83.3% (10/12; 95% CI 55-95%)** |

## Exact ideal route, and over-routing (an extra tool runs = extra latency)

| | LLM only | + safety net |
|---|---|---|
| Exact route | 58.3% (7/12; 95% CI 32-81%) | 66.7% (8/12; 95% CI 39-86%) |
| Over-routed | 16.7% (2/12; 95% CI 5-45%) | 16.7% (2/12; 95% CI 5-45%) |

## Routes changed by the safety net (1)

- **h-09** `sql` -> `both` (hybrid) - Which departments have the highest number of employees, What is the employee escalation path?

## Still missing a needed tool after the safety net (2)

- **h-05** (`sql`, needs ['rag']) - Who can solve billing questions?
- **h-08** (`sql`, needs ['rag', 'sql']) - List top 5 city has the most customers and what is the customer lookup?

## Over-routed after the safety net (2) - correct but slower

- **h-06** `both` (ideal `rag`) - How's the catalog organized?
- **h-11** `both` (ideal `rag`) - Is the downloaded albums can be refunded?
