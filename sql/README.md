# SQL Scripts for Battery SOH Flow

## Purpose
This SQL script operationalizes battery SOH prediction publishing in Azure SQL.

## Execution Order
1. `31_upsert_battery_soc_predictions.sql`

## Run Script (via Python SQL runner)
```powershell
C:/Users/hunate/Desktop/azure-data-engineering-project/.venv/Scripts/python.exe src/run_sql_script.py sql/31_upsert_battery_soc_predictions.sql
```

## Publish from Pipeline
```powershell
C:/Users/hunate/Desktop/azure-data-engineering-project/.venv/Scripts/python.exe src/battery_pipeline.py --data-path data/battery_data_with_soc.csv --publish-sql --strict-publish
```

## Validation Query
```sql
SELECT TOP 20 *
FROM analytics.battery_soc_predictions
ORDER BY PredictionTimeS DESC;
```
