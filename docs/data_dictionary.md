
# RetailPulse Data Dictionary

## 1. Project Overview

RetailPulse uses the UCI Online Retail dataset to analyze historical retail transactions, understand customer purchasing behaviour, and prepare customer-level features for analytics and machine learning.

The project uses three data layers:

1. Raw data: the original Excel dataset.
2. Cleaned transaction data: standardized transaction records with additional analytical columns.
3. Customer feature data: one row per eligible customer, containing purchasing and RFM-related features.

## 2. Original Dataset Columns

| Column | Description |
|---|---|
| InvoiceNo | Invoice identifier. Invoice numbers beginning with C indicate cancellations. |
| StockCode | Identifier for the product or stock item. |
| Description | Product description. |
| Quantity | Number of units on a transaction line. Negative values can indicate returns or adjustments. |
| InvoiceDate | Date and time of the transaction. |
| UnitPrice | Price per unit as recorded in the dataset. |
| CustomerID | Customer identifier; missing values indicate that the customer cannot be identified from that record. |
| Country | Country associated with the transaction. |

## 3. Additional Cleaned Transaction Columns

| Column | Description |
|---|---|
| IsCancellationInvoice | Boolean flag indicating whether InvoiceNo begins with C. |
| IsReturnOrNegativeQuantity | Boolean flag indicating a negative quantity. |
| IsZeroPrice | Boolean flag indicating UnitPrice equals zero. |
| IsNegativePrice | Boolean flag indicating UnitPrice is negative. |
| HasMissingCustomerID | Boolean flag indicating that CustomerID is missing. |
| HasMissingDescription | Boolean flag indicating that Description is missing. |
| LineRevenue | Quantity multiplied by UnitPrice. This is a signed transaction-line value, not audited accounting revenue. |
| InvoiceYear | Year extracted from InvoiceDate. |
| InvoiceMonth | Month number extracted from InvoiceDate. |
| InvoiceYearMonth | Year and month period derived from InvoiceDate. |
| DayOfWeek | Day of the week derived from InvoiceDate. |
| InvoiceHour | Hour extracted from InvoiceDate. |
| IsWeekend | Boolean flag indicating whether the transaction occurred on Saturday or Sunday. |

## 4. Customer-Level Feature Dataset

The customer feature dataset contains one row per customer with at least one eligible purchase.

| Feature | Description |
|---|---|
| CustomerID | Unique customer identifier. |
| FirstPurchaseDate | Date of the customer's first eligible purchase. |
| LastPurchaseDate | Date of the customer's most recent eligible purchase. |
| UniqueProductsPurchased | Number of distinct StockCode values purchased. |
| Frequency | Number of distinct eligible purchase invoices. |
| Monetary | Sum of Quantity multiplied by UnitPrice across eligible purchase lines. |
| TotalUnitsPurchased | Total quantity across eligible purchase lines. |
| AverageOrderValue | Monetary divided by Frequency. |
| RecencyDays | Number of days from the last eligible purchase to the reference date. |
| CustomerTenureDays | Days between first and last eligible purchase dates. |
| AverageItemsPerInvoice | TotalUnitsPurchased divided by Frequency. |
| ReturnOrCancellationLineCount | Number of identified-customer transaction rows with negative quantity or a cancellation invoice. This count is not subtracted from Monetary. |

## 5. Customer Feature Reference Date

The reference date is one day after the latest InvoiceDate in the cleaned dataset. For the current dataset, the reference date is 2011-12-10.

RecencyDays is calculated using the date portion of the last eligible purchase and this reference date.

## 6. Important Interpretation Notes

- Frequency counts distinct eligible purchase invoices, not individual transaction lines.
- Monetary represents the defined positive-purchase measure and does not deduct returns or cancellation lines.
- UniqueProductsPurchased counts distinct StockCode values; it does not guarantee that every code represents a distinct physical product.
- Missing CustomerID records are retained in the cleaned transaction dataset but cannot be included in customer-level RFM features.
- Inventory levels and supplier lead times are not provided by the original dataset. Any inventory simulation must be documented separately.

## 7. Source and Attribution

Dataset: UCI Online Retail

Source URL: https://archive.ics.uci.edu/dataset/352/online+retail

The dataset is distributed under the Creative Commons Attribution 4.0 International (CC BY 4.0) licence. Follow the dataset's attribution requirements when redistributing or presenting derived work.
