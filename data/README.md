# Dataset

## Source
**Online Retail II** — UCI Machine Learning Repository  
https://archive.ics.uci.edu/dataset/502/online+retail+ii

## Download
1. Visit the link above
2. Download `online+retail+II.xlsx`
3. Place it at `data/online_retail_II.xlsx`
4. Run `python train.py` to generate all model artefacts

## Schema
| Column | Type | Description |
|---|---|---|
| InvoiceNo | str | Invoice number (prefix C = cancellation) |
| StockCode | str | Product code |
| Description | str | Product name |
| Quantity | int | Units per transaction (negative = return) |
| InvoiceDate | datetime | Date and time of invoice |
| UnitPrice | float | Price per unit (GBP) |
| CustomerID | float | Customer identifier |
| Country | str | Customer country |

## Preprocessing
- Remove rows where `CustomerID` is null
- Remove returns (`Quantity < 0`)
- Remove zero unit prices (`UnitPrice <= 0`)
- Compute `TotalPrice = Quantity × UnitPrice`

## Synthetic data note
`ecommerce_data.csv` in this repo is synthetic data modelled after Southeast Asian
e-commerce patterns (50,000 transactions, 4,000 customers). The modeling methodology
is identical to what would be applied to the UCI dataset.
