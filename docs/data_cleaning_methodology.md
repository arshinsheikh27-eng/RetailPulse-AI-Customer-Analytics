
# RetailPulse Data Cleaning Methodology

## 1. Objective

The objective of data preparation is to improve data consistency, document data-quality issues, preserve meaningful transaction information, and create reliable inputs for exploratory analysis and customer feature engineering.

## 2. Source Data

The project uses the UCI Online Retail dataset.

- Original rows: 541,909
- Original columns: 8
- Original date range: 2010-12-01 to 2011-12-09

## 3. Cleaning and Standardization Steps

### Step 1: Load the source data

The original Excel file is loaded from `data/raw/Online Retail.xlsx`. The raw source file is retained separately so that the preparation process can be repeated.

### Step 2: Standardize text fields

Leading and trailing whitespace is removed from text fields. Empty text values are converted to missing values.

### Step 3: Remove exact duplicate rows

Exact duplicate rows are removed while keeping the first occurrence.

- Exact duplicate rows removed: 5,268
- Rows remaining after deduplication: 536,641

This rule assumes that fully identical rows are duplicates. It is a documented assumption because repeated-looking transaction lines can sometimes represent legitimate business records.

### Step 4: Convert data types

InvoiceDate is converted to a datetime type. Quantity and UnitPrice are converted to numeric values, and CustomerID is converted to a nullable integer type.

### Step 5: Check required fields

The script checks required transaction fields and valid dates. No rows needed to be removed for invalid required fields in the current run.

### Step 6: Preserve and flag unusual records

Potentially meaningful records are retained and flagged instead of being automatically deleted.

| Data-quality issue | Treatment |
|---|---|
| Missing CustomerID | Retained and flagged; excluded from customer-level RFM features. |
| Missing Description | Retained and flagged. |
| Cancellation invoice | Retained and identified using a flag. |
| Negative Quantity | Retained and flagged as a return or negative-quantity record. |
| Zero UnitPrice | Retained and flagged. |
| Negative UnitPrice | Retained and flagged. |

### Step 7: Add analytical columns

The cleaning script adds transaction-line value, invoice date components, and flags for cancellations, negative quantities, unusual prices, and missing values.

LineRevenue is calculated as Quantity multiplied by UnitPrice. It can be negative and should not be described as audited accounting revenue.

## 4. Cleaning Results

| Metric | Result |
|---|---:|
| Original rows | 541,909 |
| Exact duplicates removed | 5,268 |
| Rows after deduplication | 536,641 |
| Rows removed for invalid required fields | 0 |
| Rows with missing CustomerID preserved | 135,037 |
| Rows with missing Description preserved | 1,454 |
| Cancellation-invoice rows preserved | 9,251 |
| Negative-quantity rows preserved | 10,587 |
| Zero-price rows preserved | 2,510 |
| Negative-price rows preserved | 2 |
| Final rows | 536,641 |
| Final columns | 21 |

These counts describe the current preparation run. They should be regenerated if the source data or cleaning rules change.

## 5. Customer Feature Preparation

A separate customer-level dataset is created from eligible purchase records.

Eligible records must have:

- A non-missing CustomerID.
- Quantity greater than zero.
- An invoice number that does not begin with C.
- UnitPrice greater than or equal to zero.

The script calculates customer recency, frequency, monetary value, average order value, purchase dates, product variety, units purchased, and other supporting features.

The current output contains 4,339 customers and 12 columns. Validation checks confirmed unique customer rows, valid numerical ranges, consistent average calculations, and recency calculated from the documented reference date.

## 6. Limitations and Assumptions

1. Exact duplicate removal is based on full-row equality and may require review if transaction-line identifiers become available.
2. Missing customer identifiers prevent those records from contributing to customer-level features.
3. The positive-purchase monetary measure does not deduct returns or cancellations.
4. Historical RFM features describe past behaviour; they do not by themselves predict churn.
5. The source dataset does not provide actual stock on hand, supplier lead times, or replenishment costs. Any inventory optimization demonstration must identify simulated inputs clearly.

## 7. Reproducibility

The preparation workflow is implemented in:

- `src/inspect_data.py`
- `src/clean_data.py`
- `src/eda.py`
- `src/feature_engineering.py`
- `tests/validate_features.py`

Run the scripts from the project root with the project's Python virtual environment activated. Keep raw and generated CSV data out of Git according to `.gitignore`, while version-controlling the code and appropriate documentation and summaries.
