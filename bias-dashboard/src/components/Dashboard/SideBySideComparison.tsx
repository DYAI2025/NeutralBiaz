import React from 'react';
import { ArrowRightIcon } from '@heroicons/react/24/outline';

interface SideBySideComparisonProps {
  originalText: string;
  neutralizedText: string;
  overallScore: number;
}

const SideBySideComparison: React.FC<SideBySideComparisonProps> = ({
  originalText,
  neutralizedText,
  overallScore
}) => {
  const getScoreColor = (score: number) => {
    if (score < 0.3) return 'text-green-600';
    if (score < 0.7) return 'text-orange-600';
    return 'text-red-600';
  };

  const getScoreLabel = (score: number) => {
    if (score < 0.3) return 'Low Bias';
    if (score < 0.7) return 'Moderate Bias';
    return 'High Bias';
  };

  return (
    <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
      <div className="mb-6">
        <div className="flex items-center justify-between">
          <h3 className="text-lg font-semibold text-gray-900">
            Original vs. Neutralized Text
          </h3>
          <div className="text-right">
            <div className={`text-sm font-medium ${getScoreColor(overallScore)}`}>
              {getScoreLabel(overallScore)}
            </div>
            <div className="text-xs text-gray-500">
              Score: {Math.round(overallScore * 100)}%
            </div>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Original Text */}
        <div>
          <div className="mb-3">
            <h4 className="text-sm font-medium text-gray-700 flex items-center space-x-2">
              <div className="w-3 h-3 rounded-full bg-red-500"></div>
              <span>Original Text</span>
            </h4>
          </div>
          <div className="p-4 bg-red-50 rounded-lg border border-red-200">
            <p className="text-gray-700 leading-relaxed whitespace-pre-wrap">
              {originalText}
            </p>
          </div>
          <div className="mt-2 text-xs text-gray-500">
            Contains potential bias markers
          </div>
        </div>

        {/* Arrow */}
        <div className="hidden lg:flex items-center justify-center">
          <ArrowRightIcon className="h-8 w-8 text-gray-400" />
        </div>

        <div className="lg:hidden flex items-center justify-center py-2">
          <ArrowRightIcon className="h-6 w-6 text-gray-400 rotate-90" />
        </div>

        {/* Neutralized Text */}
        <div>
          <div className="mb-3">
            <h4 className="text-sm font-medium text-gray-700 flex items-center space-x-2">
              <div className="w-3 h-3 rounded-full bg-green-500"></div>
              <span>Neutralized Text</span>
            </h4>
          </div>
          <div className="p-4 bg-green-50 rounded-lg border border-green-200">
            <p className="text-gray-700 leading-relaxed whitespace-pre-wrap">
              {neutralizedText}
            </p>
          </div>
          <div className="mt-2 text-xs text-gray-500">
            Bias-neutralized version
          </div>
        </div>
      </div>

      {/* Improvement Summary */}
      <div className="mt-6 p-4 bg-blue-50 rounded-lg border border-blue-200">
        <h5 className="text-sm font-medium text-blue-900 mb-2">
          Neutralization Summary
        </h5>
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 text-sm">
          <div className="text-center">
            <div className="text-lg font-bold text-blue-700">
              {Math.max(0, Math.round((1 - overallScore) * 100))}%
            </div>
            <div className="text-blue-600">Bias Reduction</div>
          </div>
          <div className="text-center">
            <div className="text-lg font-bold text-blue-700">
              {originalText.length}
            </div>
            <div className="text-blue-600">Original Length</div>
          </div>
          <div className="text-center">
            <div className="text-lg font-bold text-blue-700">
              {neutralizedText.length}
            </div>
            <div className="text-blue-600">Neutralized Length</div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default SideBySideComparison;