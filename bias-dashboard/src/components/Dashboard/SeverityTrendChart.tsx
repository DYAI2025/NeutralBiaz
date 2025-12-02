import React from 'react';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend,
  ArcElement,
} from 'chart.js';
import { Bar, Doughnut } from 'react-chartjs-2';
import { BiasMarker } from '../../types/api';
import { getTopBiasCategories } from '../../utils/biasUtils';

ChartJS.register(
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend,
  ArcElement
);

interface SeverityTrendChartProps {
  markers: BiasMarker[];
  overallScore: number;
}

const SeverityTrendChart: React.FC<SeverityTrendChartProps> = ({
  markers,
  overallScore
}) => {
  // Prepare data for severity distribution
  const severityCounts = markers.reduce(
    (acc, marker) => {
      acc[marker.severity]++;
      return acc;
    },
    { low: 0, moderate: 0, high: 0 }
  );

  // Prepare data for category distribution
  const topCategories = getTopBiasCategories(markers, 5);

  const severityBarData = {
    labels: ['Low', 'Moderate', 'High'],
    datasets: [
      {
        label: 'Bias Markers by Severity',
        data: [severityCounts.low, severityCounts.moderate, severityCounts.high],
        backgroundColor: [
          'rgba(34, 197, 94, 0.8)',
          'rgba(249, 115, 22, 0.8)',
          'rgba(239, 68, 68, 0.8)',
        ],
        borderColor: [
          'rgba(34, 197, 94, 1)',
          'rgba(249, 115, 22, 1)',
          'rgba(239, 68, 68, 1)',
        ],
        borderWidth: 1,
      },
    ],
  };

  const categoryDoughnutData = {
    labels: topCategories.map(cat => cat.category),
    datasets: [
      {
        label: 'Bias Categories',
        data: topCategories.map(cat => cat.count),
        backgroundColor: [
          'rgba(236, 72, 153, 0.8)',
          'rgba(147, 51, 234, 0.8)',
          'rgba(59, 130, 246, 0.8)',
          'rgba(245, 158, 11, 0.8)',
          'rgba(99, 102, 241, 0.8)',
        ],
        borderColor: [
          'rgba(236, 72, 153, 1)',
          'rgba(147, 51, 234, 1)',
          'rgba(59, 130, 246, 1)',
          'rgba(245, 158, 11, 1)',
          'rgba(99, 102, 241, 1)',
        ],
        borderWidth: 1,
      },
    ],
  };

  const barOptions = {
    responsive: true,
    plugins: {
      legend: {
        display: false,
      },
      title: {
        display: true,
        text: 'Severity Distribution',
        font: {
          size: 14,
          weight: 'bold' as const,
        },
      },
    },
    scales: {
      y: {
        beginAtZero: true,
        ticks: {
          stepSize: 1,
        },
      },
    },
  };

  const doughnutOptions = {
    responsive: true,
    plugins: {
      legend: {
        position: 'right' as const,
        labels: {
          boxWidth: 12,
          padding: 15,
        },
      },
      title: {
        display: true,
        text: 'Top Bias Categories',
        font: {
          size: 14,
          weight: 'bold' as const,
        },
      },
    },
    maintainAspectRatio: false,
  };

  const getScoreColor = (score: number) => {
    if (score < 0.3) return 'text-green-600';
    if (score < 0.7) return 'text-orange-600';
    return 'text-red-600';
  };

  const getScoreBackground = (score: number) => {
    if (score < 0.3) return 'bg-green-100';
    if (score < 0.7) return 'bg-orange-100';
    return 'bg-red-100';
  };

  return (
    <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
      <div className="mb-6">
        <h3 className="text-lg font-semibold text-gray-900 mb-4">
          Bias Analysis Overview
        </h3>

        {/* Overall Score */}
        <div className={`inline-flex items-center px-4 py-2 rounded-lg ${getScoreBackground(overallScore)}`}>
          <span className="text-sm font-medium text-gray-700 mr-2">
            Overall Bias Score:
          </span>
          <span className={`text-lg font-bold ${getScoreColor(overallScore)}`}>
            {Math.round(overallScore * 100)}%
          </span>
        </div>
      </div>

      {markers.length === 0 ? (
        <div className="text-center py-12">
          <div className="text-gray-400 text-lg mb-2">No bias markers to analyze</div>
          <div className="text-gray-500 text-sm">
            Run an analysis to see bias distribution charts
          </div>
        </div>
      ) : (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Severity Bar Chart */}
          <div className="h-64">
            <Bar data={severityBarData} options={barOptions} />
          </div>

          {/* Category Doughnut Chart */}
          <div className="h-64">
            <Doughnut data={categoryDoughnutData} options={doughnutOptions} />
          </div>
        </div>
      )}

      {/* Statistics Summary */}
      {markers.length > 0 && (
        <div className="mt-6 grid grid-cols-2 sm:grid-cols-4 gap-4">
          <div className="text-center p-3 bg-gray-50 rounded-lg">
            <div className="text-lg font-bold text-gray-900">{markers.length}</div>
            <div className="text-xs text-gray-600">Total Markers</div>
          </div>
          <div className="text-center p-3 bg-red-50 rounded-lg">
            <div className="text-lg font-bold text-red-700">{severityCounts.high}</div>
            <div className="text-xs text-red-600">High Severity</div>
          </div>
          <div className="text-center p-3 bg-orange-50 rounded-lg">
            <div className="text-lg font-bold text-orange-700">{severityCounts.moderate}</div>
            <div className="text-xs text-orange-600">Moderate</div>
          </div>
          <div className="text-center p-3 bg-green-50 rounded-lg">
            <div className="text-lg font-bold text-green-700">{severityCounts.low}</div>
            <div className="text-xs text-green-600">Low Severity</div>
          </div>
        </div>
      )}
    </div>
  );
};

export default SeverityTrendChart;