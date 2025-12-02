import React from 'react';
import {
  GlobeAltIcon,
  InformationCircleIcon,
  LightBulbIcon
} from '@heroicons/react/24/outline';
import { CulturalAnalysis } from '../../types/api';

interface CulturalContextPanelProps {
  culturalAnalysis?: CulturalAnalysis;
  isLoading?: boolean;
}

const CulturalContextPanel: React.FC<CulturalContextPanelProps> = ({
  culturalAnalysis,
  isLoading = false
}) => {
  if (isLoading) {
    return (
      <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
        <div className="animate-pulse">
          <div className="flex items-center space-x-2 mb-4">
            <div className="w-5 h-5 bg-gray-300 rounded"></div>
            <div className="w-32 h-5 bg-gray-300 rounded"></div>
          </div>
          <div className="space-y-3">
            <div className="w-full h-4 bg-gray-300 rounded"></div>
            <div className="w-3/4 h-4 bg-gray-300 rounded"></div>
            <div className="w-1/2 h-4 bg-gray-300 rounded"></div>
          </div>
        </div>
      </div>
    );
  }

  if (!culturalAnalysis) {
    return (
      <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
        <div className="flex items-center space-x-2 mb-4">
          <GlobeAltIcon className="h-5 w-5 text-gray-500" />
          <h3 className="text-lg font-semibold text-gray-900">
            Cultural Context Analysis
          </h3>
        </div>
        <div className="text-center py-8">
          <GlobeAltIcon className="h-12 w-12 text-gray-300 mx-auto mb-3" />
          <p className="text-gray-500 text-sm">
            No cultural analysis available
          </p>
          <p className="text-gray-400 text-xs mt-1">
            Enable cultural context analysis in settings to see insights
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
      <div className="flex items-center space-x-2 mb-6">
        <GlobeAltIcon className="h-5 w-5 text-blue-600" />
        <h3 className="text-lg font-semibold text-gray-900">
          Cultural Context Analysis
        </h3>
      </div>

      <div className="space-y-6">
        {/* Detected Cultures */}
        {culturalAnalysis.detected_cultures && culturalAnalysis.detected_cultures.length > 0 && (
          <div>
            <div className="flex items-center space-x-2 mb-3">
              <InformationCircleIcon className="h-4 w-4 text-blue-500" />
              <h4 className="text-sm font-semibold text-gray-800">
                Detected Cultural Contexts
              </h4>
            </div>
            <div className="flex flex-wrap gap-2">
              {culturalAnalysis.detected_cultures.map((culture, index) => (
                <span
                  key={index}
                  className="inline-flex items-center px-3 py-1 rounded-full text-sm font-medium bg-blue-100 text-blue-800"
                >
                  {culture}
                </span>
              ))}
            </div>
          </div>
        )}

        {/* Cultural Bias Types */}
        {culturalAnalysis.cultural_bias_types && culturalAnalysis.cultural_bias_types.length > 0 && (
          <div>
            <div className="flex items-center space-x-2 mb-3">
              <InformationCircleIcon className="h-4 w-4 text-orange-500" />
              <h4 className="text-sm font-semibold text-gray-800">
                Cultural Bias Types
              </h4>
            </div>
            <div className="space-y-2">
              {culturalAnalysis.cultural_bias_types.map((biasType, index) => (
                <div
                  key={index}
                  className="flex items-center space-x-2 p-2 bg-orange-50 rounded border-l-2 border-orange-300"
                >
                  <div className="w-2 h-2 bg-orange-500 rounded-full"></div>
                  <span className="text-sm text-orange-800">{biasType}</span>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Recommendations */}
        {culturalAnalysis.recommendations && culturalAnalysis.recommendations.length > 0 && (
          <div>
            <div className="flex items-center space-x-2 mb-3">
              <LightBulbIcon className="h-4 w-4 text-yellow-500" />
              <h4 className="text-sm font-semibold text-gray-800">
                Cultural Sensitivity Recommendations
              </h4>
            </div>
            <div className="space-y-2">
              {culturalAnalysis.recommendations.map((recommendation, index) => (
                <div
                  key={index}
                  className="p-3 bg-yellow-50 rounded border-l-2 border-yellow-300"
                >
                  <div className="flex items-start space-x-2">
                    <div className="w-1.5 h-1.5 bg-yellow-500 rounded-full mt-2 flex-shrink-0"></div>
                    <p className="text-sm text-yellow-800 leading-relaxed">
                      {recommendation}
                    </p>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Empty state when all arrays are empty */}
        {(!culturalAnalysis.detected_cultures || culturalAnalysis.detected_cultures.length === 0) &&
         (!culturalAnalysis.cultural_bias_types || culturalAnalysis.cultural_bias_types.length === 0) &&
         (!culturalAnalysis.recommendations || culturalAnalysis.recommendations.length === 0) && (
          <div className="text-center py-6">
            <GlobeAltIcon className="h-8 w-8 text-gray-300 mx-auto mb-2" />
            <p className="text-gray-500 text-sm">
              No specific cultural context detected
            </p>
            <p className="text-gray-400 text-xs">
              This text appears to be culturally neutral
            </p>
          </div>
        )}
      </div>

      {/* Cultural Analysis Footer */}
      <div className="mt-6 pt-4 border-t border-gray-200">
        <div className="flex items-center space-x-2 text-xs text-gray-500">
          <InformationCircleIcon className="h-3 w-3" />
          <span>
            Cultural analysis is based on linguistic patterns and may not capture all nuances
          </span>
        </div>
      </div>
    </div>
  );
};

export default CulturalContextPanel;