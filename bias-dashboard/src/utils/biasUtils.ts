import type { BiasMarker } from '../types/api';

export const getBiasSeverityColor = (severity: string): string => {
  switch (severity) {
    case 'high':
      return 'bg-red-500';
    case 'moderate':
      return 'bg-orange-500';
    case 'low':
      return 'bg-green-500';
    default:
      return 'bg-gray-500';
  }
};

export const getBiasSeverityTextColor = (severity: string): string => {
  switch (severity) {
    case 'high':
      return 'text-red-700';
    case 'moderate':
      return 'text-orange-700';
    case 'low':
      return 'text-green-700';
    default:
      return 'text-gray-700';
  }
};

export const getCategoryColor = (category: string): string => {
  const colors: { [key: string]: string } = {
    'gender': 'bg-pink-500',
    'racial': 'bg-purple-500',
    'religious': 'bg-blue-500',
    'age': 'bg-yellow-500',
    'cultural': 'bg-indigo-500',
    'socioeconomic': 'bg-green-500',
    'political': 'bg-red-500',
    'other': 'bg-gray-500',
  };
  return colors[category.toLowerCase()] || colors['other'];
};

export const formatConfidence = (confidence: number): string => {
  return `${Math.round(confidence * 100)}%`;
};

// Helper function to get text segments with bias markers
export const getTextSegments = (
  text: string,
  markers: BiasMarker[]
): Array<{ text: string; isMarker: boolean; marker?: BiasMarker }> => {
  if (!markers.length) {
    return [{ text, isMarker: false }];
  }

  // Sort markers by start position
  const sortedMarkers = [...markers].sort((a, b) => a.start - b.start);
  const segments: Array<{ text: string; isMarker: boolean; marker?: BiasMarker }> = [];
  let lastIndex = 0;

  sortedMarkers.forEach((marker) => {
    // Add text before the marker
    if (marker.start > lastIndex) {
      segments.push({
        text: text.slice(lastIndex, marker.start),
        isMarker: false
      });
    }

    // Add the marker text
    segments.push({
      text: text.slice(marker.start, marker.end),
      isMarker: true,
      marker
    });

    lastIndex = marker.end;
  });

  // Add remaining text
  if (lastIndex < text.length) {
    segments.push({
      text: text.slice(lastIndex),
      isMarker: false
    });
  }

  return segments;
};

export const calculateOverallBiasScore = (markers: BiasMarker[]): number => {
  if (!markers.length) return 0;

  const totalWeight = markers.reduce((sum, marker) => {
    const severityWeight = {
      'low': 1,
      'moderate': 2,
      'high': 3,
    }[marker.severity] || 1;

    return sum + (marker.confidence * severityWeight);
  }, 0);

  return Math.min(totalWeight / markers.length / 3, 1);
};

export const groupMarkersByCategory = (markers: BiasMarker[]): Record<string, BiasMarker[]> => {
  return markers.reduce((groups, marker) => {
    const category = marker.category;
    if (!groups[category]) {
      groups[category] = [];
    }
    groups[category].push(marker);
    return groups;
  }, {} as Record<string, BiasMarker[]>);
};

export const getTopBiasCategories = (markers: BiasMarker[], limit: number = 5): Array<{
  category: string;
  count: number;
  severity: string;
}> => {
  const categoryStats = markers.reduce((stats, marker) => {
    const category = marker.category;
    if (!stats[category]) {
      stats[category] = { count: 0, severitySum: 0 };
    }
    stats[category].count++;
    stats[category].severitySum += {
      'low': 1,
      'moderate': 2,
      'high': 3,
    }[marker.severity] || 1;
    return stats;
  }, {} as Record<string, { count: number; severitySum: number }>);

  return Object.entries(categoryStats)
    .map(([category, stats]) => ({
      category,
      count: stats.count,
      severity: stats.severitySum / stats.count > 2.5 ? 'high' :
                stats.severitySum / stats.count > 1.5 ? 'moderate' : 'low'
    }))
    .sort((a, b) => b.count - a.count)
    .slice(0, limit);
};