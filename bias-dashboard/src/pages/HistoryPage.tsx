import React, { useState } from 'react';
import { Link } from 'react-router-dom';
import {
  ClockIcon,
  EyeIcon,
  ChevronLeftIcon,
  ChevronRightIcon,
  MagnifyingGlassIcon
} from '@heroicons/react/24/outline';
import { useAnalysisHistory } from '../hooks/useBiasDetection';
import { formatDistance } from 'date-fns';
import { getBiasSeverityColor } from '../utils/biasUtils';

const HistoryPage: React.FC = () => {
  const [currentPage, setCurrentPage] = useState(1);
  const [searchQuery, setSearchQuery] = useState('');
  const pageSize = 20;

  const { data: historyData, isLoading, error } = useAnalysisHistory(currentPage, pageSize);

  const filteredHistory = historyData?.items?.filter(item =>
    item.text.toLowerCase().includes(searchQuery.toLowerCase())
  ) || [];

  const handlePageChange = (newPage: number) => {
    setCurrentPage(newPage);
    window.scrollTo({ top: 0, behavior: 'smooth' });
  };

  const getScoreLabel = (score: number) => {
    if (score < 0.3) return 'Low';
    if (score < 0.7) return 'Moderate';
    return 'High';
  };

  const getScoreColor = (score: number) => {
    if (score < 0.3) return 'text-green-600 bg-green-100';
    if (score < 0.7) return 'text-orange-600 bg-orange-100';
    return 'text-red-600 bg-red-100';
  };

  if (isLoading) {
    return (
      <div className="max-w-6xl mx-auto">
        <div className="mb-8">
          <h1 className="text-3xl font-bold text-gray-900 mb-4">Analysis History</h1>
          <p className="text-gray-600">Loading your previous analyses...</p>
        </div>

        <div className="space-y-4">
          {[...Array(5)].map((_, i) => (
            <div key={i} className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
              <div className="animate-pulse">
                <div className="flex items-center justify-between mb-4">
                  <div className="w-24 h-4 bg-gray-300 rounded"></div>
                  <div className="w-16 h-4 bg-gray-300 rounded"></div>
                </div>
                <div className="w-full h-16 bg-gray-300 rounded mb-4"></div>
                <div className="flex items-center space-x-4">
                  <div className="w-20 h-4 bg-gray-300 rounded"></div>
                  <div className="w-24 h-4 bg-gray-300 rounded"></div>
                  <div className="w-16 h-4 bg-gray-300 rounded"></div>
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="max-w-6xl mx-auto">
        <div className="bg-red-50 border border-red-200 rounded-lg p-6 text-center">
          <div className="text-red-600 mb-2">Error loading history</div>
          <div className="text-red-500 text-sm">
            {error.message || 'Failed to load analysis history'}
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="max-w-6xl mx-auto">
      {/* Header */}
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-gray-900 mb-4">
          Analysis History
        </h1>
        <p className="text-gray-600">
          Browse and review your previous bias analyses
        </p>
      </div>

      {/* Search and Filters */}
      <div className="mb-6">
        <div className="flex flex-col sm:flex-row sm:items-center space-y-4 sm:space-y-0 sm:space-x-4">
          <div className="relative flex-1 max-w-md">
            <MagnifyingGlassIcon className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-gray-400" />
            <input
              type="text"
              placeholder="Search your analyses..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="w-full pl-10 pr-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-primary-500"
            />
          </div>

          {historyData && (
            <div className="flex items-center space-x-2 text-sm text-gray-600">
              <span>
                Showing {filteredHistory.length} of {historyData.total} analyses
              </span>
            </div>
          )}
        </div>
      </div>

      {/* History List */}
      {!historyData || filteredHistory.length === 0 ? (
        <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-12 text-center">
          {!historyData?.items?.length ? (
            <>
              <ClockIcon className="h-12 w-12 text-gray-300 mx-auto mb-4" />
              <h3 className="text-lg font-medium text-gray-900 mb-2">
                No analyses yet
              </h3>
              <p className="text-gray-600 mb-6">
                Start analyzing text to see your history here
              </p>
              <Link
                to="/analysis"
                className="inline-flex items-center space-x-2 bg-primary-600 text-white px-6 py-2 rounded-lg hover:bg-primary-700"
              >
                <span>Start Analyzing</span>
              </Link>
            </>
          ) : (
            <>
              <MagnifyingGlassIcon className="h-12 w-12 text-gray-300 mx-auto mb-4" />
              <h3 className="text-lg font-medium text-gray-900 mb-2">
                No matching analyses
              </h3>
              <p className="text-gray-600">
                Try adjusting your search criteria
              </p>
            </>
          )}
        </div>
      ) : (
        <div className="space-y-4">
          {filteredHistory.map((analysis) => (
            <div key={analysis.id} className="bg-white rounded-lg shadow-sm border border-gray-200 p-6 hover:shadow-md transition-shadow">
              <div className="flex items-start justify-between mb-4">
                <div className="flex items-center space-x-2">
                  <ClockIcon className="h-4 w-4 text-gray-400" />
                  <span className="text-sm text-gray-600">
                    {formatDistance(new Date(analysis.created_at), new Date(), { addSuffix: true })}
                  </span>
                </div>
                <div className="flex items-center space-x-2">
                  <span className={`inline-flex px-2 py-1 text-xs font-medium rounded-full ${getScoreColor(analysis.overall_score)}`}>
                    {getScoreLabel(analysis.overall_score)} Bias
                  </span>
                  <Link
                    to={`/analysis/${analysis.id}`}
                    className="inline-flex items-center space-x-1 text-primary-600 hover:text-primary-700 text-sm font-medium"
                  >
                    <EyeIcon className="h-4 w-4" />
                    <span>View Details</span>
                  </Link>
                </div>
              </div>

              <div className="mb-4">
                <p className="text-gray-700 leading-relaxed line-clamp-3">
                  {analysis.text}
                </p>
              </div>

              <div className="flex items-center space-x-6 text-sm text-gray-600">
                <div className="flex items-center space-x-1">
                  <div className="w-2 h-2 bg-orange-500 rounded-full"></div>
                  <span>{analysis.bias_count} bias markers</span>
                </div>
                <div className="flex items-center space-x-1">
                  <div className="w-2 h-2 bg-blue-500 rounded-full"></div>
                  <span>Score: {Math.round(analysis.overall_score * 100)}%</span>
                </div>
                <div className="flex items-center space-x-1">
                  <div className="w-2 h-2 bg-gray-400 rounded-full"></div>
                  <span>{analysis.text.length} characters</span>
                </div>
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Pagination */}
      {historyData && historyData.total > pageSize && (
        <div className="mt-8 flex items-center justify-between">
          <div className="text-sm text-gray-600">
            Showing {((currentPage - 1) * pageSize) + 1} to {Math.min(currentPage * pageSize, historyData.total)} of {historyData.total} results
          </div>

          <div className="flex items-center space-x-2">
            <button
              onClick={() => handlePageChange(currentPage - 1)}
              disabled={!historyData.has_previous}
              className="flex items-center space-x-1 px-3 py-2 border border-gray-300 rounded-lg text-sm font-medium text-gray-700 bg-white hover:bg-gray-50 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              <ChevronLeftIcon className="h-4 w-4" />
              <span>Previous</span>
            </button>

            <span className="text-sm text-gray-600">
              Page {currentPage} of {Math.ceil(historyData.total / pageSize)}
            </span>

            <button
              onClick={() => handlePageChange(currentPage + 1)}
              disabled={!historyData.has_next}
              className="flex items-center space-x-1 px-3 py-2 border border-gray-300 rounded-lg text-sm font-medium text-gray-700 bg-white hover:bg-gray-50 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              <span>Next</span>
              <ChevronRightIcon className="h-4 w-4" />
            </button>
          </div>
        </div>
      )}
    </div>
  );
};

export default HistoryPage;