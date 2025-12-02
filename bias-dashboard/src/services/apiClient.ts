import axios, { AxiosRequestConfig, AxiosResponse } from 'axios';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

// Create axios instance with default configuration
export const apiClient = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30000,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Request deduplication map
const pendingRequests = new Map<string, Promise<AxiosResponse>>();

// Generate request key for deduplication
const getRequestKey = (config: AxiosRequestConfig): string => {
  const { method = 'GET', url = '', params, data } = config;
  return `${method.toUpperCase()}:${url}:${JSON.stringify(params || {})}:${JSON.stringify(data || {})}`;
};

// Request interceptor with deduplication
apiClient.interceptors.request.use(
  (config) => {
    // Add auth token if available
    const token = localStorage.getItem('auth_token');
    if (token) {
      config.headers = config.headers || {};
      config.headers.Authorization = `Bearer ${token}`;
    }

    // Add request timestamp for debugging
    config.metadata = { startTime: new Date() };

    return config;
  },
  (error) => Promise.reject(error)
);

// Response interceptor with enhanced error handling and caching
apiClient.interceptors.response.use(
  (response: AxiosResponse) => {
    // Log response time for performance monitoring
    const startTime = response.config.metadata?.startTime;
    if (startTime) {
      const duration = new Date().getTime() - startTime.getTime();
      console.debug(`API Request to ${response.config.url} took ${duration}ms`);
    }

    return response;
  },
  (error) => {
    // Enhanced error logging
    console.error('API Error Details:', {
      url: error.config?.url,
      method: error.config?.method?.toUpperCase(),
      status: error.response?.status,
      statusText: error.response?.statusText,
      data: error.response?.data,
      message: error.message,
    });

    // Handle different error scenarios
    if (error.response?.status === 401) {
      // Unauthorized - clear auth and redirect
      localStorage.removeItem('auth_token');
      window.location.href = '/login';
    } else if (error.response?.status === 429) {
      // Rate limiting
      console.warn('Rate limit exceeded. Please try again later.');
    } else if (error.response?.status >= 500) {
      // Server errors
      console.error('Server error occurred. Please try again.');
    } else if (!error.response) {
      // Network errors
      console.error('Network error. Please check your connection.');
    }

    return Promise.reject(error);
  }
);

// Request deduplication wrapper
export const makeRequest = async <T = any>(
  requestFn: () => Promise<AxiosResponse<T>>,
  allowDuplication = false
): Promise<T> => {
  if (allowDuplication) {
    const response = await requestFn();
    return response.data;
  }

  // For GET requests, implement deduplication
  const key = getRequestKey({
    method: 'GET',
    url: requestFn.toString(), // Simple approach, could be improved
  });

  if (pendingRequests.has(key)) {
    const response = await pendingRequests.get(key)!;
    return response.data;
  }

  const requestPromise = requestFn();
  pendingRequests.set(key, requestPromise);

  try {
    const response = await requestPromise;
    return response.data;
  } finally {
    pendingRequests.delete(key);
  }
};

// Retry utility with exponential backoff
export const withRetry = async <T>(
  operation: () => Promise<T>,
  maxRetries: number = 3,
  baseDelay: number = 1000
): Promise<T> => {
  for (let attempt = 0; attempt < maxRetries; attempt++) {
    try {
      return await operation();
    } catch (error) {
      // Don't retry on the last attempt
      if (attempt === maxRetries - 1) {
        throw error;
      }

      // Don't retry client errors (4xx)
      if (axios.isAxiosError(error) && error.response?.status) {
        const status = error.response.status;
        if (status >= 400 && status < 500 && status !== 429) {
          throw error;
        }
      }

      // Calculate delay with jitter
      const delay = baseDelay * Math.pow(2, attempt) + Math.random() * 1000;
      console.warn(`Request failed, retrying in ${delay}ms... (Attempt ${attempt + 1}/${maxRetries})`);

      await new Promise(resolve => setTimeout(resolve, delay));
    }
  }

  throw new Error('Unreachable');
};

export default apiClient;