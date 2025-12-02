import { useMutation, useQuery, useQueryClient, UseQueryOptions } from '@tanstack/react-query';
import { api } from '../services/api';
import {
  AnalysisRequest,
  BatchAnalysisRequest,
  AnalysisResult,
  BatchAnalysisResult,
  HealthResponse,
  ModelsResponse,
  ConfigResponse,
  // Legacy types
  BiasDetectionRequest,
  BiasDetectionResult
} from '../types/api';

// Main analysis hooks
export const useAnalyzeText = () => {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (request: AnalysisRequest) => api.analyzeText(request),
    onSuccess: (data: AnalysisResult) => {
      // Optimistic update - add to cache
      const cacheKey = `analysis_${Date.now()}`;
      queryClient.setQueryData(['analysis', cacheKey], data);

      // Invalidate related queries
      queryClient.invalidateQueries({ queryKey: ['analysisHistory'] });
      queryClient.invalidateQueries({ queryKey: ['dashboardStats'] });
    },
    onError: (error) => {
      console.error('Analysis failed:', error);
    },
    retry: (failureCount, error) => {
      // Don't retry validation errors
      if (error && (error as any)?.response?.status === 400) {
        return false;
      }
      return failureCount < 2;
    },
    retryDelay: (attemptIndex) => Math.min(1000 * 2 ** attemptIndex, 30000),
  });
};

export const useAnalyzeBatch = () => {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (request: BatchAnalysisRequest) => api.analyzeBatch(request),
    onSuccess: () => {
      // Invalidate related queries after batch processing
      queryClient.invalidateQueries({ queryKey: ['analysisHistory'] });
      queryClient.invalidateQueries({ queryKey: ['dashboardStats'] });
    },
    retry: 1,
  });
};

// Legacy hook for backward compatibility
export const useBiasDetection = () => {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (request: BiasDetectionRequest) => api.detectBias(request),
    onSuccess: (data: BiasDetectionResult) => {
      // Cache the new analysis
      queryClient.setQueryData(['analysis', data.id], data);

      // Invalidate history to show the new analysis
      queryClient.invalidateQueries({ queryKey: ['analysisHistory'] });
      queryClient.invalidateQueries({ queryKey: ['dashboardStats'] });
    },
    onError: (error) => {
      console.error('Bias detection failed:', error);
    },
    retry: (failureCount, error) => {
      // Don't retry validation errors
      if (error && (error as any)?.response?.status === 400) {
        return false;
      }
      return failureCount < 2;
    },
  });
};

export const useAnalysis = (id: string | undefined) => {
  return useQuery({
    queryKey: ['analysis', id],
    queryFn: () => api.getAnalysis(id!),
    enabled: !!id,
  });
};

export const useAnalysisHistory = (page: number = 1, size: number = 20) => {
  return useQuery({
    queryKey: ['analysisHistory', page, size],
    queryFn: () => api.getAnalysisHistory(page, size),
    placeholderData: (previousData) => previousData,
  });
};

export const useDashboardStats = () => {
  return useQuery({
    queryKey: ['dashboardStats'],
    queryFn: api.getDashboardStats,
    refetchInterval: 5 * 60 * 1000, // Refetch every 5 minutes
  });
};

export const useBiasCategories = (options?: Partial<UseQueryOptions<string[]>>) => {
  return useQuery({
    queryKey: ['biasCategories'],
    queryFn: api.getBiasCategories,
    staleTime: 60 * 60 * 1000, // Data is fresh for 1 hour
    gcTime: 2 * 60 * 60 * 1000, // Cache for 2 hours
    retry: 2,
    ...options,
  });
};

export const useHealthCheck = () => {
  return useQuery({
    queryKey: ['healthCheck'],
    queryFn: api.healthCheck,
    refetchInterval: 30 * 1000, // Check every 30 seconds
    retry: false,
    staleTime: 10 * 1000, // Consider fresh for 10 seconds
    gcTime: 5 * 60 * 1000, // Keep in cache for 5 minutes
  });
};

// System configuration hooks
export const useModels = (options?: Partial<UseQueryOptions<ModelsResponse>>) => {
  return useQuery({
    queryKey: ['models'],
    queryFn: api.getModels,
    staleTime: 10 * 60 * 1000, // Fresh for 10 minutes
    gcTime: 30 * 60 * 1000, // Cache for 30 minutes
    retry: 2,
    ...options,
  });
};

export const useConfig = (options?: Partial<UseQueryOptions<ConfigResponse>>) => {
  return useQuery({
    queryKey: ['config'],
    queryFn: api.getConfig,
    staleTime: 15 * 60 * 1000, // Fresh for 15 minutes
    gcTime: 60 * 60 * 1000, // Cache for 1 hour
    retry: 2,
    ...options,
  });
};

// Real-time analysis status hook (for long-running analyses)
export const useAnalysisStatus = (analysisId?: string) => {
  return useQuery({
    queryKey: ['analysisStatus', analysisId],
    queryFn: () => api.getAnalysis(analysisId!),
    enabled: !!analysisId,
    refetchInterval: (data) => {
      // Stop polling if analysis is complete
      return data?.created_at ? false : 2000; // Poll every 2 seconds
    },
    retry: false,
  });
};

// Prefetch hook for improving UX
export const usePrefetchData = () => {
  const queryClient = useQueryClient();

  const prefetchConfig = () => {
    queryClient.prefetchQuery({
      queryKey: ['config'],
      queryFn: api.getConfig,
      staleTime: 15 * 60 * 1000,
    });
  };

  const prefetchModels = () => {
    queryClient.prefetchQuery({
      queryKey: ['models'],
      queryFn: api.getModels,
      staleTime: 10 * 60 * 1000,
    });
  };

  const prefetchBiasCategories = () => {
    queryClient.prefetchQuery({
      queryKey: ['biasCategories'],
      queryFn: api.getBiasCategories,
      staleTime: 60 * 60 * 1000,
    });
  };

  return {
    prefetchConfig,
    prefetchModels,
    prefetchBiasCategories,
    prefetchAll: () => {
      prefetchConfig();
      prefetchModels();
      prefetchBiasCategories();
    },
  };
};