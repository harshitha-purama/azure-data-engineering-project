IF NOT EXISTS (SELECT 1 FROM sys.schemas WHERE name = 'analytics')
    EXEC('CREATE SCHEMA analytics');
GO

IF OBJECT_ID('analytics.battery_soc_predictions', 'U') IS NULL
BEGIN
    CREATE TABLE analytics.battery_soc_predictions (
        PredictionTimeS FLOAT NOT NULL PRIMARY KEY,
        PredictedSOC FLOAT NOT NULL,
        StepSizeSeconds FLOAT NULL,
        ModelName NVARCHAR(200) NULL,
        MAE FLOAT NULL,
        R2 FLOAT NULL,
        CreatedAtUtc DATETIME2 NOT NULL,
        UpdatedAtUtc DATETIME2 NOT NULL
    );
END;
GO

CREATE OR ALTER PROCEDURE analytics.sp_upsert_battery_soc_prediction
    @PredictionTimeS FLOAT,
    @PredictedSOC FLOAT,
    @StepSizeSeconds FLOAT = NULL,
    @ModelName NVARCHAR(200) = NULL,
    @MAE FLOAT = NULL,
    @R2 FLOAT = NULL
AS
BEGIN
    SET NOCOUNT ON;

    MERGE analytics.battery_soc_predictions AS target
    USING (
        SELECT
            @PredictionTimeS AS PredictionTimeS,
            @PredictedSOC AS PredictedSOC,
            @StepSizeSeconds AS StepSizeSeconds,
            @ModelName AS ModelName,
            @MAE AS MAE,
            @R2 AS R2
    ) AS source
    ON target.PredictionTimeS = source.PredictionTimeS
    WHEN MATCHED THEN
        UPDATE SET
            PredictedSOC = source.PredictedSOC,
            StepSizeSeconds = source.StepSizeSeconds,
            ModelName = source.ModelName,
            MAE = source.MAE,
            R2 = source.R2,
            UpdatedAtUtc = SYSUTCDATETIME()
    WHEN NOT MATCHED THEN
        INSERT (
            PredictionTimeS,
            PredictedSOC,
            StepSizeSeconds,
            ModelName,
            MAE,
            R2,
            CreatedAtUtc,
            UpdatedAtUtc
        )
        VALUES (
            source.PredictionTimeS,
            source.PredictedSOC,
            source.StepSizeSeconds,
            source.ModelName,
            source.MAE,
            source.R2,
            SYSUTCDATETIME(),
            SYSUTCDATETIME()
        );
END;
GO
