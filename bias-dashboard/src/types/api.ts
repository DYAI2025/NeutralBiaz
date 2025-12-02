// Enums matching FastAPI backend
export enum BiasType {
  GENDER = "gender",
  RACIAL = "racial",
  AGE = "age",
  RELIGIOUS = "religious",
  POLITICAL = "political",
  SOCIOECONOMIC = "socioeconomic",
  CULTURAL = "cultural",
  DISABILITY = "disability",
  LGBTQ = "lgbtq",
  UNKNOWN = "unknown"
}

export enum BiasLevel {
  LOW = "low",
  MEDIUM = "medium",
  HIGH = "high",
  CRITICAL = "critical"
}

export enum AnalysisMode {
  FAST = "fast",
  ACCURATE = "accurate",
  COMPREHENSIVE = "comprehensive"
}

export enum CulturalProfile {
  NEUTRAL = "neutral",
  WESTERN = "western",
  EASTERN = "eastern",
  GLOBAL = "global"
}

// Request interfaces matching FastAPI backend
export interface AnalysisRequest {
  text: string;
  mode?: AnalysisMode;
  cultural_profile?: CulturalProfile;
  language?: string;
  bias_types?: BiasType[];
  confidence_threshold?: number;
  include_suggestions?: boolean;
}

export interface BatchAnalysisRequest {
  texts: string[];
  mode?: AnalysisMode;
  cultural_profile?: CulturalProfile;
  language?: string;
  bias_types?: BiasType[];
  confidence_threshold?: number;
  include_suggestions?: boolean;
}

// Detection result interfaces matching FastAPI backend
export interface BiasDetection {
  type: BiasType;
  level: BiasLevel;
  confidence: number;
  description: string;
  affected_text: string;
  start_position: number;
  end_position: number;
  suggestions: string[];
}

export interface AnalysisResult {
  text: string;
  language: string;
  overall_bias_score: number;
  bias_level: BiasLevel;
  detections: BiasDetection[];
  neutralized_text?: string;
  processing_time: number;
  model_version: string;
}

export interface BatchAnalysisResult {
  results: AnalysisResult[];
  total_processed: number;
  total_processing_time: number;
  summary: Record<string, number>;
}

// Legacy interfaces for compatibility (will be phased out)
export interface BiasDetectionRequest {
  text: string;
  options?: {
    include_cultural_context?: boolean;
    include_severity_analysis?: boolean;
    language?: string;
  };
}

export interface BiasMarker {
  id: string;
  start: number;
  end: number;
  text: string;
  category: string;
  subcategory: string;
  severity: 'low' | 'moderate' | 'high';
  confidence: number;
  suggestions: string[];
  cultural_context?: string;
}

export interface BiasDetectionResult {
  id: string;
  original_text: string;
  neutralized_text: string;
  markers: BiasMarker[];
  overall_score: number;
  cultural_analysis?: CulturalAnalysis;
  processing_time: number;
  created_at: string;
}

export interface CulturalAnalysis {
  detected_cultures: string[];
  cultural_bias_types: string[];
  recommendations: string[];
}

export interface BiasCategory {
  name: string;
  subcategories: string[];
  color: string;
  description: string;
}

export interface ApiResponse<T> {
  data: T;
  message?: string;
  status: 'success' | 'error';
}

export interface PaginatedResponse<T> {
  items: T[];
  total: number;
  page: number;
  size: number;
  has_next: boolean;
  has_previous: boolean;
}

export interface ErrorResponse {
  error: string;
  details?: string;
  code?: string;
}

export interface AnalysisHistory {
  id: string;
  text: string;
  bias_count: number;
  overall_score: number;
  created_at: string;
}

// Health check response
export interface HealthResponse {
  status: string;
  timestamp: string;
  version: string;
  uptime: number;
  dependencies: Record<string, string>;
}

// Model information
export interface ModelInfo {
  name: string;
  version: string;
  description: string;
  supported_languages: string[];
  supported_bias_types: BiasType[];
  accuracy?: number;
  is_default: boolean;
  loaded: boolean;
}

export interface ModelsResponse {
  models: ModelInfo[];
  default_model: string;
  total_models: number;
}

// Configuration response
export interface ConfigResponse {
  app_name: string;
  app_version: string;
  environment: string;
  supported_languages: string[];
  supported_bias_types: BiasType[];
  cultural_profiles: CulturalProfile[];
  analysis_modes: AnalysisMode[];
  default_settings: Record<string, string | number | boolean>;
}

export interface DashboardStats {
  total_analyses: number;
  average_bias_score: number;
  most_common_bias: string;
  analyses_today: number;
  improvement_rate: number;
}