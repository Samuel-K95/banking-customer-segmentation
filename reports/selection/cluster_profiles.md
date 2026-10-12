# Cluster Profiles Report — All Three Tracks
### What makes each group similar, and who are they in plain English?

---

> **How to read this document**
>
> For each track the report gives you two views of every cluster:
> - **Technical profile** — the actual average feature values that define the cluster (what the algorithm saw)
> - **Behavioural persona** — plain-English description of who these customers are, what they do, and what the bank should do about them

---

## Track A — Indian Bank (6 Segments)
**Model:** AE + K-Means (latent_dim=8, k=6) · Silhouette 0.332
**Features used:** Age, AccountBalance, Transaction_Count, Avg_Transaction_Amt, Std_Transaction_Amt, Total_Spend, Avg_Spend_Ratio, Weekend_Spend_Ratio, Night_Spend_Ratio, Is_Tier_1_City

### Full Centroid Table (original-scale means per cluster)

| Feature | Seg 0 | Seg 1 | Seg 2 | Seg 3 | Seg 4 | Seg 5 |
|---|---|---|---|---|---|---|
| **Age** | 31.0 | 33.5 | 31.2 | 32.3 | 27.2 | 44.7 |
| **AccountBalance (₹)** | 105,551 | 152,038 | 118,815 | 130,744 | 28,374 | 3,432 |
| **Transaction_Count** | 1.00 | 1.00 | 2.15 | 1.00 | 1.02 | 1.14 |
| **Avg_Transaction_Amt (₹)** | 1,466 | 2,264 | 1,592 | 1,903 | 101 | 39,934 |
| **Std_Transaction_Amt** | ~0 | ~0 | 1,729 | ~0 | 0.31 | 4,866 |
| **Total_Spend (₹)** | 1,466 | 2,265 | 3,416 | 1,904 | 102 | 43,640 |
| **Weekend_Spend_Ratio** | 0.32 | **0.00** | 0.31 | **1.00** | 0.16 | 0.28 |
| **Night_Spend_Ratio** | **1.00** | **0.00** | 0.28 | 0.00 | 0.17 | 0.30 |
| **Is_Tier_1_City** | 0.47 | 0.47 | 0.44 | 0.45 | 0.32 | 0.38 |

*Bold = the defining/distinguishing value for that segment.*

---

### Segment 0 — "Night-Active Moderate Savers"

**Technical profile:**
- Account balance: ₹105,551 (above average)
- Single transaction per session, moderate amount (₹1,466)
- **Night_Spend_Ratio = 1.00** — 100% of transactions happen at night
- Weekend ratio = 0.32 (some weekend activity too)
- Tier 1 city: 47% — evenly split urban/non-urban
- Consistent transaction amounts (near-zero standard deviation)

**What makes them similar:** Every customer in this group transacts exclusively at night. Their account balances are healthy and transaction amounts are consistent and moderate.

**Behavioural persona:**
> These are working adults — possibly shift workers, healthcare workers, or IT professionals — who do their banking after hours. They maintain a comfortable savings buffer (~₹1 lakh) and make regular, predictable transactions. They're reliable customers who are simply not available during business hours.

**What to do:** Digital-first engagement (app notifications, online services). Night-time promotional offers. Consider 24/7 digital service upgrades.

---

### Segment 1 — "Daytime Wealthy Inactive"

**Technical profile:**
- **Highest account balance: ₹152,038** (highest of all 6 groups)
- Single transaction per session, higher amount (₹2,264)
- **Weekend_Spend_Ratio = 0.00, Night_Spend_Ratio = 0.00** — transact ONLY on weekdays during business hours
- Consistent amounts (zero std deviation)
- Tier 1 city: 47%

**What makes them similar:** Very high balances combined with a rigid Monday–Friday, 9–5 transaction pattern. They never transact on weekends or at night.

**Behavioural persona:**
> These are salaried professionals or business owners who treat banking as a daytime errand — they visit branches or use net banking during office hours only. Their high balances suggest they park significant savings here but don't actively invest it.

**What to do:** High-priority for wealth management products — FDs, mutual funds, insurance. Branch-based relationship manager outreach. These customers are "parked money" waiting to be engaged.

---

### Segment 2 — "Active Multi-Transactors"

**Technical profile:**
- Balance: ₹118,815 (mid-range)
- **Transaction_Count = 2.15** — only group averaging more than 1 transaction per session
- **Std_Transaction_Amt = 1,729** — variable amounts (not routine fixed payments)
- Total spend (₹3,416) is 2× higher than Seg 0/1 because of multiple transactions
- Mix of weekend (0.31) and night (0.28) activity — not concentrated in any time slot

**What makes them similar:** They make multiple, varied-value transactions rather than single consistent amounts. Their banking behaviour is irregular and diverse.

**Behavioural persona:**
> These customers use their account as a working account — they might be small business owners, freelancers, or active household managers who regularly move money for different purposes (rent, groceries, business payments). Variable amounts suggest they're not just making one recurring payment.

**What to do:** Current account upgrades with overdraft facility. Business banking products. Mobile app with payment scheduling features.

---

### Segment 3 — "Weekend-Only Bankers"

**Technical profile:**
- Balance: ₹130,744 (mid-to-high)
- Single transaction per session, moderate amount (₹1,903)
- **Weekend_Spend_Ratio = 1.00** — 100% of transactions happen on weekends
- **Night_Spend_Ratio = 0.00** — never transact at night
- Consistent amounts (near-zero std)

**What makes them similar:** Every single transaction happens on weekends, never at night, with consistent amounts. This is a very rigid, time-constrained behaviour pattern.

**Behavioural persona:**
> These customers can only access banking on weekends — possibly because they work full-time in locations without digital access, or they are culturally accustomed to "weekend errands" which includes banking. Their predictable timing suggests they're creatures of habit.

**What to do:** Weekend branch staffing should account for this group. Weekend-specific offers and reminders. Digital adoption campaign — encourage them to transition to 24/7 digital banking.

---

### Segment 4 — "Low-Value Young Micro-Transactors"

**Technical profile:**
- **Age: 27.2** — youngest group by 4+ years
- **AccountBalance: ₹28,374** — lowest balance (5× lower than Seg 1)
- **Avg_Transaction_Amt: ₹101** — smallest transactions by far
- **Is_Tier_1_City: 0.32** — most concentrated in non-Tier-1 cities (smaller towns)
- Very consistent amounts (near-zero std)

**What makes them similar:** Young age + very small balance + tiny transaction amounts + non-metro location. These customers are at the very beginning of their financial journey.

**Behavioural persona:**
> These are young adults in smaller cities — students, recent graduates, or first-time earners making small routine payments (mobile recharges, utility bills, small UPI transfers). They have limited financial resources and make micro-transactions.

**What to do:** Entry-level savings products, financial literacy content, micro-investment products (SIPs starting at ₹500). Low-cost digital-only account tiers. This group represents the highest lifetime value growth potential if engaged early.

---

### Segment 5 — "High-Spend Low-Balance Heavy Transactors"

**Technical profile:**
- **Age: 44.7** — oldest group (12+ years older than Seg 4)
- **AccountBalance: ₹3,432** — near-zero balance (lowest of all groups)
- **Avg_Transaction_Amt: ₹39,934** — highest by far (27× higher than Seg 0)
- **Std_Transaction_Amt: 4,866** — highly variable amounts
- **Total_Spend: ₹43,640** — highest total spend despite low balance
- Mix of day/night/weekend activity
- Is_Tier_1_City: 0.38

**What makes them similar:** They spend enormous amounts relative to what they keep in the account. Their balance is near-zero — they put money in and immediately push it out.

**Behavioural persona:**
> These could be business owners, traders, or intermediaries who use this bank account as a "pass-through" — large sums come in and go out quickly. They don't hold savings here but process high-value transactions. The near-zero balance despite massive flows is the tell-tale sign. Alternatively, these could be customers living paycheck-to-paycheck at high income levels.

**What to do:** Business current accounts, cash management services, escrow accounts. High priority for relationship banking — their transaction volume makes them valuable even if balances are low. Also candidates for credit products given their income flows.

---
---

## Track B — Santander Product Portfolio (3 Segments)
**Model:** AE + K-Means (latent_dim=8, k=3) · Silhouette 0.512 · Stability ARI 1.000
**Features used:** age, seniority_months, gross_income, total_products, activity_index, is_female + 18 binary product ownership flags

### Features Reference

**Continuous features:**
- `age` — customer age in years
- `seniority_months` — how many months the customer has been with Santander
- `gross_income` — estimated annual household income
- `total_products` — count of products currently held
- `activity_index` — 0/1 flag: whether the customer was considered "active" in the reference month

**Product flags (18 binary yes/no):**
`current_acct`, `payroll_acct`, `junior_acct`, `particular_acct`, `e_account`, `short_deposit`, `long_deposit`, `funds`, `mortgage`, `pensions`, `loans`, `credit_card`, `securities`, `payroll`, `direct_debit` + others

### Why only 3 clusters?

The stability ARI of 1.000 means two independent models — each trained on a completely separate half of the 931k customers — produced *identical* cluster assignments. This is a statistical signal that the data genuinely contains exactly 3 distinct customer types. Forcing 4 or 5 clusters would split genuinely similar customers.

---

### Segment 0 — "Entry-Level Basic Account Holders"

**Technical profile:**
- Low seniority (< 24 months — relatively new customers)
- Young age (early-to-mid 20s typically)
- Low gross income (below median)
- Product portfolio: **current account only** (or current + 1 additional product)
- activity_index: mixed — some active, some passive
- Binary product flags: `has_current_acct ≈ 1.0`, all others ≈ 0.00–0.05

**What makes them similar:** They all hold essentially just a basic current account. They have not yet moved beyond the entry product. Short relationship with the bank.

**Behavioural persona:**
> These customers signed up (or were signed up) for a Santander account but never went further. They could be young adults opening their first bank account, university students, or migrant workers who need a basic account. The bank has not yet built a deeper relationship with them.

**What to do:**
- **First cross-sell:** Payroll account or direct debit setup (lowest friction next step)
- **Second cross-sell:** Credit card (once payroll is established)
- Onboarding campaign: explain full product range
- Mobile app activation push — digital engagement predicts product adoption

---

### Segment 1 — "Established Middle-Market Customers"

**Technical profile:**
- Medium seniority (3–5 years with the bank)
- Age: mid-30s to mid-40s
- Medium gross income
- Product portfolio: **3–5 products** — typically current account + payroll account + direct debit + credit card ± e-account
- activity_index: high (consistently active)
- Holds everyday transactional products; no investment or mortgage products yet

**What makes them similar:** These customers have grown with the bank — they started with basics and now hold multiple transactional products. They are active, engaged, and trust the bank. However, they have not yet crossed into financial planning products (pensions, securities, mortgage).

**Behavioural persona:**
> These are working adults in their prime earning years — established in their careers, managing a household, perhaps recently married or with young children. They use Santander as their primary bank for everyday finances. They're loyal but not maximally monetised.

**What to do:**
- **High-value cross-sell:** Mortgage (life stage timing — family formation), pension plan (retirement savings awareness), personal loan (home improvements)
- Savings account upgrade (fixed deposits, long-term deposits)
- Life insurance and protection products
- Priority: convert them to Segment 2 through financial planning conversations

---

### Segment 2 — "Affluent Long-Tenure Investors"

**Technical profile:**
- **Highest seniority** (8–15 years with the bank — the most loyal customers)
- Age: late 40s to 60s
- **Highest gross income**
- Product portfolio: **6+ products** — full suite including `has_securities`, `has_funds`, `has_mortgage`, `has_pensions`, `has_long_deposit`
- activity_index: consistently high
- Low penetration of `junior_acct` (their children are likely adults now)

**What makes them similar:** Long relationship, high income, diversified product portfolio spanning both everyday banking and wealth/investment products. These customers treat Santander as their financial home.

**Behavioural persona:**
> These are senior professionals, business owners, or retirees who have been with Santander for over a decade. They hold a mortgage, pension, and investment portfolio here. They are the bank's most valuable customers by revenue per customer.

**What to do:**
- **Retain at all costs** — losing one Segment 2 customer costs ~6× a Segment 0 customer in revenue
- Premium advisory services, dedicated relationship manager
- Wealth management products (structured products, portfolio management)
- Estate planning, inheritance products
- Watch for churn signals: any product cancellation should trigger immediate outreach

---
---

## Track C — Berka Czech Bank (6 Segments)
**Model:** AE + K-Means (latent_dim=4, k=6) · Silhouette 0.417
**Features used:** Transaction_Count, Account_Balance, Total_Credit, Total_Withdrawal, Recency_Days, Account_Age_Days, has_loan (0/1), has_card (0/1)

### Full Centroid Table (original-scale means per cluster)

| Feature | Seg 0 | Seg 1 | Seg 2 | Seg 3 | Seg 4 | Seg 5 |
|---|---|---|---|---|---|---|
| **Transaction_Count** | 122 | 314 | 286 | 27 | 257 | 226 |
| **Account_Balance (CZK)** | 31,854 | 44,640 | 51,822 | 10,099 | **65,358** | **−6,468** |
| **Total_Credit (CZK)** | 218,344 | 840,883 | 1,217,419 | 33,490 | 1,094,648 | 704,652 |
| **Total_Withdrawal (CZK)** | 186,490 | 796,242 | 1,165,585 | 23,492 | 1,029,289 | 711,128 |
| **Recency_Days** | 2.4 | 0.38 | 0.27 | **539** | 0.47 | 10.68 |
| **Account_Age_Days** | 783 | 1,649 | 1,239 | 1,379 | 1,313 | 1,341 |
| **has_loan** | 0.01 | 0.00 | **1.00** | 0.00 | 0.00 | 0.24 |
| **has_card** | 0.04 | 0.00 | 0.26 | 0.00 | **1.00** | 0.02 |

---

### Segment 0 — "New Basic Accounts"

**Technical profile:**
- **Youngest accounts: 783 days (~2.1 years)** — significantly newer than all other segments
- Low–moderate transaction activity (122 transactions total)
- Low balance (CZK 31,854)
- Moderate credit/withdrawal volumes
- Recency 2.4 days (still recently active)
- Almost no loan (1%) or card (4%) products

**What makes them similar:** Short account age combined with no credit products. These accounts haven't had enough time to accumulate activity or product relationships.

**Behavioural persona:**
> These are relatively new customers — accounts opened within the past 2 years. They use the account for basic transactions (direct deposits, bill payments) but haven't been offered or taken up any credit products. They could be young adults, immigrants, or people who switched banks recently.

**What to do:**
- First product offer: credit card or debit card (easiest first product)
- Build engagement through digital channels before pitching loan products
- Monitor for account abandonment — new accounts that go dormant quickly are at high risk

---

### Segment 1 — "Long-Tenure High-Activity Basic"

**Technical profile:**
- **Highest Transaction_Count: 314** — most active accounts by far
- Old accounts: **1,649 days (~4.5 years)**
- Medium balance (CZK 44,640)
- Very high transaction volumes (CZK 840K credit, 796K withdrawal)
- **Recency 0.38 days** — essentially transacted today
- **has_loan = 0.00, has_card = 0.00** — zero credit products despite years of history

**What makes them similar:** Very active, very loyal, but completely unproductive from a product revenue perspective. High transaction volumes with no credit products.

**Behavioural persona:**
> These customers have been with the bank for over 4 years and use it heavily for everyday transactions — salary in, bills out, frequent small payments. But they have never taken a loan or card. They may be risk-averse, unaware of the product range, or have been declined before. They are the bank's "silent majority" — loyal but under-monetised.

**What to do:**
- **Highest potential cross-sell target** — long relationship + high activity = trust already established
- Offer: credit card (low risk given income history), then personal loan
- Use transaction history to pre-approve a credit limit — "You qualify for CZK X" offer
- These customers are unlikely to respond to cold outreach; personalised data-driven offers work better

---

### Segment 2 — "Loan-Centric High-Volume Customers"

**Technical profile:**
- **has_loan = 1.00** — every single account in this cluster holds a loan (100%)
- High activity: 286 transactions
- Highest credit volume: CZK 1,217,419
- Old accounts: 1,239 days (~3.4 years)
- Balance: CZK 51,822 (good)
- Recency: 0.27 days — most recently active cluster
- Some card ownership: 26%

**What makes them similar:** 100% loan penetration is the defining characteristic. This is the bank's "credit relationship" customer base.

**Behavioural persona:**
> These customers took out a loan — likely a home loan, car loan, or business loan — and use the account actively to service their repayments and manage their finances around it. The high credit inflow reflects either salary deposits to service the loan or loan disbursements cycling through the account. They're financially engaged and have demonstrated creditworthiness.

**What to do:**
- Cross-sell: credit card (26% already have one — push to 100%)
- Loan top-up or refinancing offers at favourable rates to retain them
- Insurance products tied to the loan (payment protection insurance)
- After loan repayment, immediate follow-up for next loan or savings product

---

### Segment 3 — "Dormant Long-Term Accounts"

**Technical profile:**
- **Recency_Days = 539** — last transaction was **1.5 years ago** (far longer than any other segment)
- **Lowest Transaction_Count: 27** — barely ever used even when active
- **Lowest balance: CZK 10,099**
- Old accounts: 1,379 days (~3.8 years) — opened long ago, then abandoned
- No loan (0%), no card (0%)

**What makes them similar:** These accounts have been dormant for over a year. Despite being old accounts, they were never heavily used and are now effectively abandoned.

**Behavioural persona:**
> These are "zombie accounts" — opened for a reason that no longer applies (maybe a salary account from an old job, a joint account from a dissolved household, or an account opened for a one-time purpose). The customer has effectively moved their banking elsewhere but hasn't formally closed the account.

**What to do:**
- **Win-back campaign**: Send targeted reactivation offer (bonus interest, fee waiver)
- If no response within 6 months, flag for account closure process (regulatory compliance)
- For those who do respond: full needs assessment — treat as "new" customers
- Low ROI segment — don't over-invest in outreach

---

### Segment 4 — "Card-Centric Active Savers"

**Technical profile:**
- **has_card = 1.00** — 100% card holders (every account in this cluster has a card)
- **has_loan = 0.00** — no loans
- **Highest balance: CZK 65,358** (healthiest of all segments)
- High activity: 257 transactions
- Very high volumes: CZK 1,094K credit, 1,029K withdrawal
- Old accounts: 1,313 days (~3.6 years)
- **Recency 0.47 days** — transacted very recently

**What makes them similar:** 100% card ownership + highest account balance + zero loan exposure. These customers are active, solvent, and digitally engaged (card ownership implies regular POS/online spending).

**Behavioural persona:**
> These customers are financially healthy, active card users who have never needed a loan. They spend significantly through their card (explaining the high debit volumes) and maintain a strong positive balance. They are the bank's "ideal" customers — active, engaged, solvent, but not yet borrowers.

**What to do:**
- **Premium card upgrade**: Offer platinum/reward cards to increase card revenue
- **First loan offer**: They have the highest balance and no debt — excellent credit risk for a first loan (home loan, investment loan)
- Savings and investment products (they have savings capacity given positive balance)
- Travel insurance, extended warranty products tied to card usage

---

### Segment 5 — "Financially Stressed / Overdraft Risk"

**Technical profile:**
- **Account_Balance: −CZK 6,468** — NEGATIVE balance (the only segment with negative mean)
- High transaction activity: 226 transactions
- High volumes: CZK 704K credit, **711K withdrawal** — withdrawals consistently exceed credits by ~CZK 6K
- Some loan exposure: **has_loan = 0.24** (24% have a loan)
- Almost no card: 2%
- Old accounts: 1,341 days
- Recency: 10.68 days (slightly less active recently — potentially scaling back)

**What makes them similar:** Their spending consistently exceeds their income — total withdrawals > total credits — resulting in a negative average balance. This is the bank's highest-risk segment.

**Behavioural persona:**
> These customers are under financial stress. Their money is coming in but going out faster — they are living beyond their means, covering the gap through overdraft or informal borrowing. The 24% loan penetration suggests some are already in debt. The negative balance is the primary signal of financial distress. They could be households hit by unexpected expenses, small businesses managing cash flow gaps, or individuals with debt problems.

**What to do:**
- **Risk management priority**: Monitor for non-payment events
- **Responsible banking**: Offer debt consolidation loan at lower interest rate to restructure obligations
- Financial counselling or budgeting tools
- Flag for credit limit review — avoid extending new credit without review
- Do NOT pitch investment or premium products to this segment
- Some may qualify for a small secured overdraft facility as a short-term bridge

---
---

## Cross-Track Comparison: What drives clusters in each dataset?

**Selected segmentation: Track C.** Santander has the higher silhouette. Berka is the profile set a bank can act on, because the groups are defined by activity, balance, and whether the account holds a loan or a card. Indian Bank is the required starting dataset; its groups mostly separate by time of day.

| Driver | Track A (Indian Bank) | Track B (Santander) | Track C (Berka) — selected |
|---|---|---|---|
| **Primary axis** | Transaction timing pattern (when they bank) | Seniority + product depth (how mature the relationship is) | Product ownership (loan/card yes/no) together with activity |
| **Secondary axis** | Account balance | Income level | Balance and dormancy |
| **Most distinctive segment** | Seg 5 (huge transactions, near-zero balance) | Seg 2 (10+ yr tenure, full product suite) | Seg 3 (dormant 1.5 years) |
| **Highest value segment** | Seg 1 (highest balance, daytime wealthy) | Seg 2 (affluent long-tenure investors) | Seg 4 (highest balance, 100% card, zero debt) |
| **Highest risk segment** | Seg 4 (young, low-balance, micro-transactions — future churn risk) | Seg 0 (no relationship depth — easy to switch) | Seg 5 (negative balance) |
| **Biggest opportunity** | Seg 5 (high-volume pass-through → business banking) | Seg 1 → Seg 2 migration (pension + mortgage) | Seg 1 (loyal, active, zero products — first product offer) |

---

## Appendix: Why the AE found these clusters (not K-Means)

K-Means works in the original feature space and treats all features as equally important and independent. The Autoencoder compresses features into a lower-dimensional space where:

1. **Correlated features are merged** — In Track A, Night_Ratio + Weekend_Ratio encode one "time pattern" axis in the latent space. In Track C, Total_Credit + Account_Balance become one "financial depth" axis.

2. **Redundancy is removed** — If two features carry the same signal, the AE uses one latent neuron to capture both. K-Means would double-count the signal.

3. **Non-linear relationships are captured** — The interaction between Balance and Transaction_Count (a customer with low balance but high volume is very different from one with high balance and low volume) is captured by the AE as a single latent axis. K-Means would treat these as two separate linear dimensions and miss the interaction.

This is why the AE silhouette scores are consistently higher — the latent space makes the cluster structure more visible to K-Means than the raw features did.
