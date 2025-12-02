import React, { useMemo } from 'react';
import { BiasDetection, BiasMarker } from '../../types/api';
import { ExclamationTriangleIcon } from '@heroicons/react/24/outline';

// Enhanced props to support both new and legacy formats
interface BiasHeatmapProps {
  text: string;
  markers?: BiasMarker[]; // Legacy format
  detections?: BiasDetection[]; // New format
  onMarkerClick?: (marker: BiasMarker | BiasDetection) => void;
  isLoading?: boolean;
  error?: Error;
}

const BiasHeatmap: React.FC<BiasHeatmapProps> = ({
  text,
  markers = [],
  detections = [],
  onMarkerClick,
  isLoading = false,
  error
}) => {
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
  const handleMarkerClick = (marker: BiasMarker) => {
    if (onMarkerClick) {
      onMarkerClick(marker);
    }
  };

  const getSeverityColor = (severity: string) => {
    switch (severity.toLowerCase()) {
      case 'high':
      case 'critical':
        return 'bg-red-100 border-red-300 text-red-800';
      case 'medium':
      case 'moderate':
        return 'bg-orange-100 border-orange-300 text-orange-800';
      case 'low':
        return 'bg-yellow-100 border-yellow-300 text-yellow-800';
      default:
        return 'bg-gray-100 border-gray-300 text-gray-800';
    }
  };

  const renderHighlightedText = () => {
    if (!processedMarkers.length) {
      return <span className="text-gray-700 leading-relaxed">{text}</span>;
    }

    // Sort markers by start position and handle overlaps
    const sortedMarkers = [...processedMarkers]
      .sort((a, b) => a.start - b.start)
      .filter((marker, index, arr) => {
        // Remove overlapping markers (keep the first one)
        if (index === 0) return true;
        const prev = arr[index - 1];
        return marker.start >= prev.end;
      });

    const elements: React.ReactNode[] = [];
    let lastIndex = 0;

    sortedMarkers.forEach((marker, index) => {
      // Add text before the marker
      if (marker.start > lastIndex) {
        elements.push(
          <span key={`text-${index}`} className="text-gray-700">
            {text.slice(lastIndex, marker.start)}
          </span>
        );
      }

      // Add the highlighted marker with proper styling
      const severityClass = getSeverityColor(marker.severity);
      elements.push(
        <span
          key={`marker-${marker.id}`}
          className={`inline-block px-1 py-0.5 rounded border cursor-pointer hover:shadow-sm transition-all duration-200 ${severityClass}`}
          title={`${marker.category}: ${marker.subcategory} (${Math.round(marker.confidence * 100)}% confidence)`}
          onClick={() => handleMarkerClick(marker)}
        >
          {text.slice(marker.start, marker.end)}
        </span>
      );

      lastIndex = marker.end;
    });

    // Add remaining text
    if (lastIndex < text.length) {
      elements.push(
        <span key="text-end" className="text-gray-700">
          {text.slice(lastIndex)}
        </span>
      );
    }

    return elements;
  };

  if (error) {
    return (
      <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
        <div className="flex items-center justify-center p-8 text-red-600">
          <ExclamationTriangleIcon className="h-8 w-8 mr-3" />
          <div>
            <h3 className="text-lg font-medium">Analysis Failed</h3>
            <p className="text-sm text-gray-600 mt-1">
              {error.message || 'Failed to analyze text for bias'}
            </p>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
      <div className="mb-4">
        <div className="flex items-center justify-between mb-2">
          <h3 className="text-lg font-semibold text-gray-900">
            Text Analysis
          </h3>
          {isLoading && (
            <div className="flex items-center space-x-2 text-sm text-gray-500">
              <div className="animate-spin w-4 h-4 border-2 border-primary-500 border-t-transparent rounded-full"></div>
              <span>Analyzing...</span>
            </div>
          )}
        </div>

        <div className="flex flex-wrap items-center gap-4 text-sm text-gray-600">
          <span>
            {processedMarkers.length} bias marker{processedMarkers.length !== 1 ? 's' : ''} detected
          </span>

          {!isLoading && (
            <div className="flex items-center space-x-3">
              <div className="flex items-center space-x-1">
                <div className="w-3 h-3 rounded bg-red-200 border border-red-300"></div>
                <span>High/Critical</span>
              </div>
              <div className="flex items-center space-x-1">
                <div className="w-3 h-3 rounded bg-orange-200 border border-orange-300"></div>
                <span>Medium</span>
              </div>
              <div className="flex items-center space-x-1">
                <div className="w-3 h-3 rounded bg-yellow-200 border border-yellow-300"></div>
                <span>Low</span>
              </div>
            </div>
          )}
        </div>
      </div>

      <div className="prose max-w-none">
        <div className="p-4 bg-gray-50 rounded-lg leading-relaxed text-base min-h-[100px]">
          {isLoading ? (
            <div className="flex items-center justify-center py-8">
              <div className="text-gray-500">Analyzing text for bias patterns...</div>
            </div>
          ) : (
            renderHighlightedText()
          )}
        </div>
      </div>

      {!isLoading && processedMarkers.length > 0 && (
        <div className="mt-4 text-sm text-gray-500">
          Click on highlighted text to view bias details and suggestions
        </div>
      )}
    </div>
  );
};

export default BiasHeatmap;

// Custom CSS for bias highlighting (add to your global CSS)
/*
.bias-highlight {
  @apply px-1 py-0.5 rounded border cursor-pointer transition-all duration-200;
}

.bias-highlight.high,
.bias-highlight.critical {
  @apply bg-red-100 border-red-300 text-red-800 hover:shadow-sm;
}

.bias-highlight.medium,
.bias-highlight.moderate {
  @apply bg-orange-100 border-orange-300 text-orange-800 hover:shadow-sm;
}

.bias-highlight.low {
  @apply bg-yellow-100 border-yellow-300 text-yellow-800 hover:shadow-sm;
}
*/