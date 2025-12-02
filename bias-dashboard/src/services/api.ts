import axios, { AxiosResponse, AxiosError } from 'axios';
import {
  AnalysisRequest,
  BatchAnalysisRequest,
  AnalysisResult,
  BatchAnalysisResult,
  HealthResponse,
  ModelInfo,
  ModelsResponse,
  ConfigResponse,
  ApiResponse,
  PaginatedResponse,
  AnalysisHistory,
  DashboardStats,
  ErrorResponse,
  // Legacy types for compatibility
  BiasDetectionRequest,
  BiasDetectionResult
} from '../types/api';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

// Create axios instance with default config
const apiClient = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30000,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Request interceptor
apiClient.interceptors.request.use(
  (config) => {
    // Add auth token if available
    const token = localStorage.getItem('auth_token');
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
  },
  (error) => Promise.reject(error)
);

// Response interceptor with enhanced error handling
apiClient.interceptors.response.use(
  (response: AxiosResponse) => response,
  (error: AxiosError<ErrorResponse>) => {
    // Log error for debugging
    console.error('API Error:', {
      url: error.config?.url,
      method: error.config?.method,
      status: error.response?.status,
      data: error.response?.data
    });

    if (error.response?.status === 401) {
      // Handle unauthorized - redirect to login
      localStorage.removeItem('auth_token');
      window.location.href = '/login';
    } else if (error.response?.status === 429) {
      // Handle rate limiting
      console.warn('Rate limit exceeded, please try again later');
    } else if (error.response?.status >= 500) {
      // Server errors
      console.error('Server error occurred');
    } else if (!error.response) {
      // Network errors
      console.error('Network error occurred');
    }

    return Promise.reject(error);
  }
);

// Utility function for handling API errors with retry logic
const withRetry = async <T>(
  operation: () => Promise<T>,
  maxRetries: number = 3,
  delay: number = 1000
): Promise<T> => {
  for (let i = 0; i < maxRetries; i++) {
    try {
      return await operation();
    } catch (error) {
      if (i === maxRetries - 1) throw error;

      // Only retry on network errors or 5xx status codes
      if (axios.isAxiosError(error)) {
        const status = error.response?.status;
        if (status && status >= 400 && status < 500) {
          throw error; // Don't retry client errors
        }
      }

      // Exponential backoff
      await new Promise(resolve => setTimeout(resolve, delay * Math.pow(2, i)));
    }
  }
  throw new Error('Unreachable');
};

// API methods
export const api = {
  // Main analysis endpoints (FastAPI backend)
  async analyzeText(request: AnalysisRequest): Promise<AnalysisResult> {
    return withRetry(async () => {
      const response = await apiClient.post<AnalysisResult>('/api/v1/analyze', request);
      return response.data;
    });
  },

  async analyzeBatch(request: BatchAnalysisRequest): Promise<BatchAnalysisResult> {
    return withRetry(async () => {
      const response = await apiClient.post<BatchAnalysisResult>('/api/v1/analyze/batch', request);
      return response.data;
    });
  },

  // System endpoints
  async healthCheck(): Promise<HealthResponse> {
    const response = await apiClient.get<HealthResponse>('/api/v1/health');
    return response.data;
  },

  async getModels(): Promise<ModelsResponse> {
    return withRetry(async () => {
      const response = await apiClient.get<ModelsResponse>('/api/v1/models');
      return response.data;
    });
  },

  async getConfig(): Promise<ConfigResponse> {
    return withRetry(async () => {
      const response = await apiClient.get<ConfigResponse>('/api/v1/config');
      return response.data;
    });
  },

  // Legacy endpoints (for backward compatibility)
  async detectBias(request: BiasDetectionRequest): Promise<BiasDetectionResult> {
    // Convert legacy request to new format
    const analysisRequest: AnalysisRequest = {
      text: request.text,
      language: request.options?.language || 'auto',
      include_suggestions: true
    };

    try {
      const result = await this.analyzeText(analysisRequest);

      // Convert new format back to legacy format
      return {
        id: `analysis_${Date.now()}`,
        original_text: result.text,
        neutralized_text: result.neutralized_text || '',
        markers: result.detections.map((detection, index) => ({
          id: `marker_${index}`,
          start: detection.start_position,
          end: detection.end_position,
          text: detection.affected_text,
          category: detection.type,
          subcategory: detection.level,
          severity: detection.level as 'low' | 'moderate' | 'high',
          confidence: detection.confidence,
          suggestions: detection.suggestions,
          cultural_context: detection.description
        })),
        overall_score: result.overall_bias_score,
        processing_time: result.processing_time,
        created_at: new Date().toISOString(),
        cultural_analysis: {
          detected_cultures: [],
          cultural_bias_types: result.detections.map(d => d.type),
          recommendations: result.detections.flatMap(d => d.suggestions)
        }
      };
    } catch (error) {
      console.error('Error in detectBias:', error);
      throw error;
    }
  },

  // Get analysis by ID
  async getAnalysis(id: string): Promise<BiasDetectionResult> {
    const response = await apiClient.get<ApiResponse<BiasDetectionResult>>(
      `/api/v1/analysis/${id}`
    );
    return response.data.data;
  },

  // Get analysis history
  async getAnalysisHistory(
    page: number = 1,
    size: number = 20
  ): Promise<PaginatedResponse<AnalysisHistory>> {
    const response = await apiClient.get<PaginatedResponse<AnalysisHistory>>(
      '/api/v1/analysis/history',
      { params: { page, size } }
    );
    return response.data;
  },

  // Get dashboard statistics
  async getDashboardStats(): Promise<DashboardStats> {
    const response = await apiClient.get<ApiResponse<DashboardStats>>(
      '/api/v1/dashboard/stats'
    );
    return response.data.data;
  },

  // Get available bias categories
  async getBiasCategories(): Promise<string[]> {
    try {
      const config = await this.getConfig();
      return config.supported_bias_types;
    } catch {
      // Fallback to legacy endpoint or default values
      return ['gender', 'racial', 'age', 'religious', 'political', 'socioeconomic', 'cultural'];
    }
  }
};

export default api;