# Source library

This folder contains the evidence base for the scivol literature and product
audit. It was assembled on 25 August 2026.

## Indexes

- `source_index.csv` is the machine-readable index.
- `source_index.md` is the same index in readable form.
- `selected_open_documents/` contains local copies of a small set of openly
  accessible or official documents used repeatedly in the audit.

The index is generated from every external link in the synthesis and topic notes. It records
the link text, year where the citation supplies one, DOI where the URL exposes
one, original URL, and the topic notes that use the source. The index includes
primary papers, authoritative regulatory material, and official software
documentation. Topic notes favor primary sources over secondary summaries.

The local PDF set is selective. Most publishers expose stable article or DOI
pages but do not grant redistribution rights for the final PDF. The index links
to those pages instead of copying restricted files. Each downloaded document
retains its original terms and copyright.

## Selected local documents

| File | Original source |
|---|---|
| `arch_documentation.pdf` | [Official arch documentation](https://arch.readthedocs.io/) |
| `rugarch_manual.pdf` | [CRAN rugarch manual](https://cran.r-project.org/web/packages/rugarch/rugarch.pdf) |
| `rmgarch_manual.pdf` | [CRAN rmgarch manual](https://cran.r-project.org/web/packages/rmgarch/rmgarch.pdf) |
| `bcbs_market_risk_2019.pdf` | [Basel minimum capital requirements for market risk](https://www.bis.org/bcbs/publ/d457.htm) |
| `bcbs_stress_testing_2018.pdf` | [Basel stress testing principles](https://www.bis.org/bcbs/publ/d450.htm) |
| `boj_rough_volatility_survey_2024.pdf` | [Bank of Japan survey of rough volatility](https://www.imes.boj.or.jp/research/papers/english/24-E-06.html) |
| `christoffersen_interval_forecasts_1998.pdf` | [Author-hosted copy of Evaluating Interval Forecasts](https://users.ssc.wisc.edu/~behansen/718/Christoffersen1998.pdf) |
| `engle_manganelli_caviar_2004.pdf` | [Author-hosted CAViaR paper](https://www.simonemanganelli.org/Simone/Research.html) |

## Rebuild

Run `python localdev/research_source_index.py` from the repository root after
adding or changing a topic note. The script deduplicates exact URLs and merges
topic references.
