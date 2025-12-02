import React, { useState } from 'react';
import {
  DocumentTextIcon,
  PaperAirplaneIcon,
  ExclamationTriangleIcon
} from '@heroicons/react/24/outline';
import { useBiasDetection } from '../hooks/useBiasDetection';
import { BiasDetectionRequest, BiasMarker } from '../types/api';
import BiasHeatmap from '../components/Dashboard/BiasHeatmap';
import MarkerExplorer from '../components/Dashboard/MarkerExplorer';
import SideBySideComparison from '../components/Dashboard/SideBySideComparison';
import SeverityTrendChart from '../components/Dashboard/SeverityTrendChart';
import CulturalContextPanel from '../components/Dashboard/CulturalContextPanel';

const AnalysisPage: React.FC = () => {
  const [inputText, setInputText] = useState('');
  const [includeCulturalContext, setIncludeCulturalContext] = useState(true);
  const [includeSeverityAnalysis, setIncludeSeverityAnalysis] = useState(true);
  const [selectedMarker, setSelectedMarker] = useState<BiasMarker>();

  const { mutate: detectBias, data: result, isPending: isLoading, error } = useBiasDetection();

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!inputText.trim()) return;

    const request: BiasDetectionRequest = {
      text: inputText.trim(),
      options: {
        include_cultural_context: includeCulturalContext,
        include_severity_analysis: includeSeverityAnalysis,
        language: 'en',
      },
    };

    detectBias(request);
  };

  const handleMarkerSelect = (marker: BiasMarker) => {
    setSelectedMarker(marker);
  };

  const exampleTexts = [
    "The candidate should be able to handle the pressures of this role, as it can be quite demanding for someone in their position.",
    "Our team is looking for a cultural fit who can work well with the guys in engineering.",
    "We need someone who is articulate and professional to represent our company."
  ];

  const loadExample = (text: string) => {
    setInputText(text);
  };

  return (
    <div className="max-w-7xl mx-auto space-y-8">
      {/* Header */}
      <div className="text-center">
        <h1 className="text-3xl font-bold text-gray-900 mb-4">
          Bias Analysis Tool
        </h1>
        <p className="text-gray-600 max-w-2xl mx-auto">
          Paste your text below to analyze it for potential bias and receive
          neutralization suggestions with cultural context insights.
        </p>
      </div>

      {/* Input Form */}
      <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
        <form onSubmit={handleSubmit} className="space-y-6">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Text to Analyze
            </label>
            <textarea
              value={inputText}
              onChange={(e) => setInputText(e.target.value)}
              placeholder="Paste your text here for bias analysis..."
              className="w-full h-40 p-4 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-primary-500 resize-none"
              disabled={isLoading}
            />
            <div className="flex justify-between items-center mt-2">
              <div className="text-sm text-gray-500">
                {inputText.length} characters
              </div>
              <div className="text-sm text-gray-500">
                Minimum 10 characters required
              </div>
            </div>
          </div>

          {/* Example texts */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Quick Examples
            </label>
            <div className="grid grid-cols-1 gap-2">
              {exampleTexts.map((text, index) => (
                <button
                  key={index}
                  type="button"
                  onClick={() => loadExample(text)}
                  className="text-left p-3 text-sm bg-gray-50 hover:bg-gray-100 rounded border border-gray-200 transition-colors"
                  disabled={isLoading}
                >
                  "{text}"
                </button>
              ))}
            </div>
          </div>

          {/* Options */}
          <div className="flex flex-wrap gap-4">
            <label className="flex items-center space-x-2">
              <input
                type="checkbox"
                checked={includeCulturalContext}
                onChange={(e) => setIncludeCulturalContext(e.target.checked)}
                className="rounded border-gray-300 text-primary-600 focus:ring-primary-500"
                disabled={isLoading}
              />
              <span className="text-sm text-gray-700">Include Cultural Context</span>
            </label>
            <label className="flex items-center space-x-2">
              <input
                type="checkbox"
                checked={includeSeverityAnalysis}
                onChange={(e) => setIncludeSeverityAnalysis(e.target.checked)}
                className="rounded border-gray-300 text-primary-600 focus:ring-primary-500"
                disabled={isLoading}
              />
              <span className="text-sm text-gray-700">Include Severity Analysis</span>
            </label>
          </div>

          {/* Submit Button */}
          <button
            type="submit"
            disabled={isLoading || inputText.trim().length < 10}
            className="w-full md:w-auto flex items-center justify-center space-x-2 bg-primary-600 text-white px-8 py-3 rounded-lg font-medium hover:bg-primary-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
          >
            {isLoading ? (
              <>
                <div className="animate-spin w-4 h-4 border-2 border-white border-t-transparent rounded-full"></div>
                <span>Analyzing...</span>
              </>
            ) : (
              <>
                <PaperAirplaneIcon className="h-4 w-4" />
                <span>Analyze Text</span>
              </>
            )}
          </button>
        </form>

        {/* Error Message */}
        {error && (
          <div className="mt-4 p-4 bg-red-50 border border-red-200 rounded-lg">
            <div className="flex items-center space-x-2">
              <ExclamationTriangleIcon className="h-5 w-5 text-red-500" />
              <span className="text-sm font-medium text-red-700">
                Analysis Error
              </span>
            </div>
            <p className="text-sm text-red-600 mt-1">
              {(error as any)?.response?.data?.error || error?.message || 'Failed to analyze text. Please try again.'}
            </p>
          </div>
        )}
      </div>

      {/* Results */}
      {result && (
        <div className="space-y-8">
          {/* Bias Heatmap */}
          <BiasHeatmap
            text={result.original_text}
            markers={result.markers}
            onMarkerClick={handleMarkerSelect}
          />

          {/* Side by Side Comparison */}
          <SideBySideComparison
            originalText={result.original_text}
            neutralizedText={result.neutralized_text}
            overallScore={result.overall_score}
          />

          <div className="grid grid-cols-1 xl:grid-cols-2 gap-8">
            {/* Marker Explorer */}
            <MarkerExplorer
              markers={result.markers}
              selectedMarker={selectedMarker}
              onMarkerSelect={handleMarkerSelect}
            />

            {/* Cultural Context Panel */}
            <CulturalContextPanel
              culturalAnalysis={result.cultural_analysis}
            />
          </div>

          {/* Severity Trend Chart */}
          <SeverityTrendChart
            markers={result.markers}
            overallScore={result.overall_score}
          />

          {/* Analysis Summary */}
          <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
            <h3 className="text-lg font-semibold text-gray-900 mb-4">
              Analysis Summary
            </h3>
            <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
              <div className="text-center p-4 bg-gray-50 rounded-lg">
                <div className="text-2xl font-bold text-gray-900">
                  {result.markers.length}
                </div>
                <div className="text-sm text-gray-600">Bias Markers</div>
              </div>
              <div className="text-center p-4 bg-gray-50 rounded-lg">
                <div className="text-2xl font-bold text-gray-900">
                  {Math.round(result.overall_score * 100)}%
                </div>
                <div className="text-sm text-gray-600">Overall Score</div>
              </div>
              <div className="text-center p-4 bg-gray-50 rounded-lg">
                <div className="text-2xl font-bold text-gray-900">
                  {result.processing_time.toFixed(2)}s
                </div>
                <div className="text-sm text-gray-600">Processing Time</div>
              </div>
              <div className="text-center p-4 bg-gray-50 rounded-lg">
                <div className="text-2xl font-bold text-gray-900">
                  {result.original_text.length}
                </div>
                <div className="text-sm text-gray-600">Characters</div>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default AnalysisPage;

// Performance optimization: Preload the heavy components
const prefetchComponents = () => {
  import('../components/Dashboard/BiasHeatmap');
  import('../components/Dashboard/MarkerExplorer');
  import('../components/Dashboard/SideBySideComparison');
  import('../components/Dashboard/SeverityTrendChart');
  import('../components/Dashboard/CulturalContextPanel');
};

// Prefetch when the module loads
if (typeof window !== 'undefined') {
  setTimeout(prefetchComponents, 100);
}