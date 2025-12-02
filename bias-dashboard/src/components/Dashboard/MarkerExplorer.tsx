import React, { useState, useMemo } from 'react';
import { BiasMarker, BiasDetection, BiasType, BiasLevel } from '../../types/api';
import {
  FunnelIcon,
  ExclamationTriangleIcon,
  LightBulbIcon,
  ChartBarIcon
} from '@heroicons/react/24/outline';

// Enhanced interface to support both new and legacy formats
interface MarkerExplorerProps {
  markers?: BiasMarker[]; // Legacy format
  detections?: BiasDetection[]; // New format
  selectedMarker?: BiasMarker | BiasDetection;
  onMarkerSelect?: (marker: BiasMarker | BiasDetection) => void;
  isLoading?: boolean;
  error?: Error;
}

// Utility functions (inline to avoid dependency on external utils)
const getBiasSeverityColor = (severity: string) => {
  switch (severity.toLowerCase()) {
    case 'high':
    case 'critical':
      return 'bg-red-500';
    case 'medium':
    case 'moderate':
      return 'bg-orange-500';
    case 'low':
      return 'bg-yellow-500';
    default:
      return 'bg-gray-500';
  }
};

const getCategoryColor = (category: string) => {
  const colors: Record<string, string> = {
    gender: 'bg-pink-500',
    racial: 'bg-purple-500',
    age: 'bg-blue-500',
    religious: 'bg-green-500',
    political: 'bg-red-500',
    socioeconomic: 'bg-indigo-500',
    cultural: 'bg-teal-500',
    disability: 'bg-orange-500',
    lgbtq: 'bg-rainbow-500',
    unknown: 'bg-gray-500'
  };
  return colors[category.toLowerCase()] || 'bg-gray-500';
};

const formatConfidence = (confidence: number) => {
  return `${Math.round(confidence * 100)}%`;
};

const MarkerExplorer: React.FC<MarkerExplorerProps> = ({
  markers = [],
  detections = [],
  selectedMarker,
  onMarkerSelect,
  isLoading = false,
  error
}) => {
  const [filterCategory, setFilterCategory] = useState<string>('all');
  const [filterSeverity, setFilterSeverity] = useState<string>('all');
  const [sortBy, setSortBy] = useState<'position' | 'confidence' | 'severity'>('position');

  // Convert detections to markers format for compatibility
  const processedMarkers = useMemo(() => {
    if (detections.length > 0) {
      return detections.map((detection, index) => ({
        id: `detection_${index}`,
        start: detection.start_position,
        end: detection.end_position,
        text: detection.affected_text,
        category: detection.type,
        subcategory: detection.level,
        severity: detection.level as 'low' | 'moderate' | 'high',
        confidence: detection.confidence,
        suggestions: detection.suggestions,
        cultural_context: detection.description
      } as BiasMarker));
    }
    return markers;
  }, [detections, markers]);

  // Group markers by category
  const groupedMarkers = useMemo(() => {
    return processedMarkers.reduce((groups, marker) => {
      const category = marker.category || 'unknown';
      if (!groups[category]) {
        groups[category] = [];
      }
      groups[category].push(marker);
      return groups;
    }, {} as Record<string, BiasMarker[]>);
  }, [processedMarkers]);

  const categories = Object.keys(groupedMarkers);

  const filteredAndSortedMarkers = useMemo(() => {
    let filtered = processedMarkers.filter(marker => {
      const categoryMatch = filterCategory === 'all' || marker.category === filterCategory;
      const severityMatch = filterSeverity === 'all' || marker.severity === filterSeverity;
      return categoryMatch && severityMatch;
    });

    // Sort markers
    filtered.sort((a, b) => {
      switch (sortBy) {
        case 'confidence':
          return b.confidence - a.confidence;
        case 'severity':
          const severityOrder = { high: 3, critical: 3, moderate: 2, medium: 2, low: 1 };
          const aSeverity = severityOrder[a.severity as keyof typeof severityOrder] || 0;
          const bSeverity = severityOrder[b.severity as keyof typeof severityOrder] || 0;
          return bSeverity - aSeverity;
        case 'position':
        default:
          return a.start - b.start;
      }
    });

    return filtered;
  }, [processedMarkers, filterCategory, filterSeverity, sortBy]);

  const handleMarkerClick = (marker: BiasMarker | BiasDetection) => {
    if (onMarkerSelect) {
      onMarkerSelect(marker);
    }
  };

  if (error) {
    return (
      <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
        <div className="flex items-center justify-center p-8 text-red-600">
          <ExclamationTriangleIcon className="h-8 w-8 mr-3" />
          <div>
            <h3 className="text-lg font-medium">Failed to Load Markers</h3>
            <p className="text-sm text-gray-600 mt-1">
              {error.message || 'Unable to load bias markers'}
            </p>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
      <div className="mb-6">
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center space-x-2">
            <ChartBarIcon className="h-5 w-5 text-gray-500" />
            <h3 className="text-lg font-semibold text-gray-900">
              Bias Markers ({filteredAndSortedMarkers.length})
            </h3>
            {isLoading && (
              <div className="animate-spin w-4 h-4 border-2 border-primary-500 border-t-transparent rounded-full ml-2"></div>
            )}
          </div>
        </div>

        {/* Filters and Sort Controls */}
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Category
            </label>
            <select
              value={filterCategory}
              onChange={(e) => setFilterCategory(e.target.value)}
              className="block w-full px-3 py-2 border border-gray-300 rounded-md text-sm focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-primary-500"
              disabled={isLoading}
            >
              <option value="all">All Categories</option>
              {categories.map(category => (
                <option key={category} value={category}>
                  {category.charAt(0).toUpperCase() + category.slice(1)} ({groupedMarkers[category].length})
                </option>
              ))}
            </select>
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Severity
            </label>
            <select
              value={filterSeverity}
              onChange={(e) => setFilterSeverity(e.target.value)}
              className="block w-full px-3 py-2 border border-gray-300 rounded-md text-sm focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-primary-500"
              disabled={isLoading}
            >
              <option value="all">All Severities</option>
              <option value="high">High/Critical</option>
              <option value="moderate">Moderate</option>
              <option value="low">Low</option>
            </select>
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Sort By
            </label>
            <select
              value={sortBy}
              onChange={(e) => setSortBy(e.target.value as 'position' | 'confidence' | 'severity')}
              className="block w-full px-3 py-2 border border-gray-300 rounded-md text-sm focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-primary-500"
              disabled={isLoading}
            >
              <option value="position">Position</option>
              <option value="confidence">Confidence</option>
              <option value="severity">Severity</option>
            </select>
          </div>
        </div>
      </div>

      {/* Markers List */}
      <div className="space-y-3 max-h-96 overflow-y-auto custom-scrollbar">
        {isLoading ? (
          <div className="space-y-3">
            {[1, 2, 3].map(i => (
              <div key={i} className="animate-pulse p-4 rounded-lg border border-gray-200">
                <div className="flex items-center space-x-2 mb-2">
                  <div className="w-3 h-3 bg-gray-300 rounded-full"></div>
                  <div className="h-4 bg-gray-300 rounded w-1/2"></div>
                </div>
                <div className="h-3 bg-gray-300 rounded w-3/4 mb-2"></div>
                <div className="h-3 bg-gray-300 rounded w-1/2"></div>
              </div>
            ))}
          </div>
        ) : filteredAndSortedMarkers.length === 0 ? (
          <div className="text-center py-8 text-gray-500">
            {processedMarkers.length === 0
              ? 'No bias markers detected'
              : 'No markers match the current filters'
            }
          </div>
        ) : (
          filteredAndSortedMarkers.map((marker) => (
            <div
              key={marker.id}
              className={`p-4 rounded-lg border cursor-pointer transition-all ${
                selectedMarker?.id === marker.id
                  ? 'border-primary-300 bg-primary-50 shadow-md'
                  : 'border-gray-200 bg-gray-50 hover:border-gray-300 hover:bg-gray-100'
              }`}
              onClick={() => handleMarkerClick(marker)}
            >
              <div className="flex items-start justify-between mb-2">
                <div className="flex items-center space-x-2">
                  <span className={`inline-block w-3 h-3 rounded-full ${getBiasSeverityColor(marker.severity)}`} />
                  <span className="font-medium text-gray-900 line-clamp-1">
                    "{marker.text}"
                  </span>
                </div>
                <div className="text-right flex flex-col">
                  <div className="text-xs text-gray-500">
                    {formatConfidence(marker.confidence)} confidence
                  </div>
                  <div className="text-xs text-gray-400">
                    Position {marker.start}-{marker.end}
                  </div>
                </div>
              </div>

              <div className="flex items-center flex-wrap gap-2 mb-2">
                <span className={`inline-flex px-2 py-1 text-xs font-medium rounded-full text-white ${getCategoryColor(marker.category)}`}>
                  {marker.category?.charAt(0).toUpperCase()}{marker.category?.slice(1)}
                </span>
                {marker.subcategory && (
                  <span className="text-sm text-gray-600 bg-gray-100 px-2 py-1 rounded-full">
                    {marker.subcategory}
                  </span>
                )}
                <span className={`inline-flex px-2 py-1 text-xs font-medium rounded-full capitalize ${
                  marker.severity === 'high' || marker.severity === 'critical' ? 'bg-red-100 text-red-800' :
                  marker.severity === 'moderate' || marker.severity === 'medium' ? 'bg-orange-100 text-orange-800' :
                  'bg-yellow-100 text-yellow-800'
                }`}>
                  {marker.severity}
                </span>
              </div>

              {marker.suggestions && marker.suggestions.length > 0 && (
                <div className="mt-2">
                  <div className="flex items-center space-x-1 mb-1">
                    <LightBulbIcon className="h-4 w-4 text-yellow-500" />
                    <span className="text-xs font-medium text-gray-700">
                      Suggestions:
                    </span>
                  </div>
                  <ul className="text-xs text-gray-600 space-y-1">
                    {marker.suggestions.slice(0, 2).map((suggestion, index) => (
                      <li key={index} className="flex items-start space-x-1">
                        <span>•</span>
                        <span>{suggestion}</span>
                      </li>
                    ))}
                    {marker.suggestions.length > 2 && (
                      <li className="text-gray-500">
                        +{marker.suggestions.length - 2} more suggestions
                      </li>
                    )}
                  </ul>
                </div>
              )}

              {marker.cultural_context && (
                <div className="mt-2 p-2 bg-blue-50 rounded border-l-2 border-blue-300">
                  <div className="flex items-center space-x-1 mb-1">
                    <ExclamationTriangleIcon className="h-4 w-4 text-blue-500" />
                    <span className="text-xs font-medium text-blue-700">
                      Cultural Context:
                    </span>
                  </div>
                  <p className="text-xs text-blue-600">
                    {marker.cultural_context}
                  </p>
                </div>
              )}
            </div>
          ))
        )}
      </div>
    </div>
  );
};

export default MarkerExplorer;