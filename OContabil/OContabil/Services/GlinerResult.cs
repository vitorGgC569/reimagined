using System;
using System.Collections.Generic;
using System.Text.Json;
using System.Text.Json.Serialization;

namespace OContabil.Services;

/// <summary>
/// Resultado da extração GLiNER com metadados de confiança.
/// </summary>
public class GlinerResult
{
    public bool Success { get; set; }
    public string Model { get; set; } = string.Empty;
    public string? OcrText { get; set; }
    public int TextLength { get; set; }
    
    [JsonPropertyName("avg_confidence")]
    public double AvgConfidence { get; set; }
    
    [JsonPropertyName("entity_count")]
    public int EntityCount { get; set; }
    public string? Error { get; set; }
    public string? Note { get; set; }
    public JsonElement? Extraction { get; set; }
    public double? ThresholdUsed { get; set; }
    public Dictionary<string, EntityPrediction>? RawEntities { get; set; }
}

/// <summary>
/// Predição individual de entidade com localização no texto.
/// </summary>
public class EntityPrediction
{
    public string Text { get; set; } = string.Empty;
    public string Label { get; set; } = string.Empty;
    public int Start { get; set; }
    public int End { get; set; }
    public float Confidence { get; set; }
}
